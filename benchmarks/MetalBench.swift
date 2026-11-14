import Foundation
import Metal
import Dispatch

struct BenchmarkCase: Decodable {
    let operation: String
    let size: Int
    let variant: String
    let precision: String
    let warmup: Int
    let repeats: Int
}

struct BenchmarkRequest: Decodable { let cases: [BenchmarkCase] }

struct Measurement: Encodable {
    let operation: String
    let size: Int
    let device: String
    let precision: String
    let variant: String
    let median_ms: Double
    let operations: Int64
    let gflops: Double
    let modeled_bytes: Int64
    let estimated_gb_per_s: Double
    let arithmetic_intensity: Double
    let max_abs_error: Double
    let worker_threads: Int
}

func fail(_ message: String) -> Never {
    fputs("MetalBench: \(message)\n", stderr)
    exit(1)
}

func makeBuffer(device: MTLDevice, values: [Float]) -> MTLBuffer {
    let length = values.count * MemoryLayout<Float>.stride
    return values.withUnsafeBytes { raw in
        device.makeBuffer(bytes: raw.baseAddress!, length: length, options: .storageModeShared)!
    }
}

func execute(_ queue: MTLCommandQueue, _ encode: (MTLComputeCommandEncoder) -> Void) {
    guard let command = queue.makeCommandBuffer(),
          let encoder = command.makeComputeCommandEncoder() else {
        fail("could not create a command buffer")
    }
    encode(encoder)
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    if command.status != .completed {
        fail(command.error?.localizedDescription ?? "Metal command failed")
    }
}

func outputValues(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
}

func maxError(_ actual: [Float], _ reference: [Float]) -> Double {
    zip(actual, reference).reduce(0.0) { max($0, abs(Double($1.0 - $1.1))) }
}

func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    let middle = sorted.count / 2
    return sorted.count.isMultiple(of: 2)
        ? (sorted[middle - 1] + sorted[middle]) / 2.0
        : sorted[middle]
}

func timeKernel(warmup: Int, repeats: Int, run: () -> Void) -> Double {
    for _ in 0..<warmup { run() }
    var samples = [Double]()
    for _ in 0..<repeats {
        let start = DispatchTime.now().uptimeNanoseconds
        run()
        let end = DispatchTime.now().uptimeNanoseconds
        samples.append(Double(end - start) / 1_000_000.0)
    }
    return median(samples)
}

func matrixValues(_ n: Int, seed: Int) -> [Float] {
    (0..<(n * n)).map { i in Float((i * seed % 101) - 50) / 50.0 }
}

func vectorValues(_ n: Int, seed: Int) -> [Float] {
    (0..<n).map { i in Float((i * seed % 101) - 50) / 50.0 }
}

func runMatrixCase(_ item: BenchmarkCase, device: MTLDevice, queue: MTLCommandQueue,
                   pipelines: [String: MTLComputePipelineState]) -> Measurement {
    let n = item.size
    let count = n * n
    let av = matrixValues(n, seed: 17)
    let bv = matrixValues(n, seed: 29)
    let a = makeBuffer(device: device, values: av)
    let b = makeBuffer(device: device, values: bv)
    let elementBytes = 4
    guard let output = device.makeBuffer(length: count * elementBytes,
                                        options: .storageModeShared) else {
        fail("could not allocate GEMM output")
    }
    let baseName = item.variant == "optimized" ? "matmul_tiled" : "matmul_naive"
    let pipelineName = baseName
    guard let pipeline = pipelines[pipelineName] else { fail("missing pipeline \(pipelineName)") }
    var dimension = UInt32(n)
    let threads = MTLSize(width: 16, height: 16, depth: 1)
    let groups = MTLSize(width: (n + 15) / 16, height: (n + 15) / 16, depth: 1)
    let encode = { (encoder: MTLComputeCommandEncoder) in
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(b, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
    }
    let elapsed = timeKernel(warmup: item.warmup, repeats: item.repeats) {
        execute(queue, encode)
    }
    let actual = outputValues(output, count: count)
    var reference = [Float](repeating: 0, count: count)
    for row in 0..<n {
        for column in 0..<n {
            var sum: Float = 0
            for k in 0..<n { sum += av[row * n + k] * bv[k * n + column] }
            reference[row * n + column] = sum
        }
    }
    let error = maxError(actual, reference)
    let operations = Int64(2) * Int64(n) * Int64(n) * Int64(n)
    let elementsRead: Int64
    if item.variant == "naive" {
        elementsRead = 2 * Int64(n) * Int64(n) * Int64(n)
    } else {
        let tiles = Int64((n + 15) / 16)
        elementsRead = 2 * Int64(n) * Int64(n) * tiles
    }
    let bytes = (elementsRead + Int64(count)) * Int64(elementBytes)
    let seconds = elapsed / 1_000.0
    return Measurement(operation: item.operation, size: n, device: "mps",
        precision: item.precision, variant: item.variant, median_ms: elapsed,
        operations: operations, gflops: Double(operations) / seconds / 1e9,
        modeled_bytes: bytes, estimated_gb_per_s: Double(bytes) / seconds / 1e9,
        arithmetic_intensity: Double(operations) / Double(bytes), max_abs_error: error,
        worker_threads: 0)
}

func runConvCase(_ item: BenchmarkCase, device: MTLDevice, queue: MTLCommandQueue,
                 pipelines: [String: MTLComputePipelineState]) -> Measurement {
    let height = item.size
    let width = height
    let inChannels = 8
    let outChannels = 16
    let inputValues = vectorValues(inChannels * height * width, seed: 17)
    let weightValues = vectorValues(outChannels * inChannels * 3 * 3, seed: 29)
    let input = makeBuffer(device: device, values: inputValues)
    let weights = makeBuffer(device: device, values: weightValues)
    let outputCount = outChannels * height * width
    guard let output = device.makeBuffer(length: outputCount * MemoryLayout<Float>.stride,
                                         options: .storageModeShared) else {
        fail("could not allocate convolution output")
    }
    var dimension = UInt32(height)
    let optimized = item.variant == "optimized"
    let workItems = optimized ? outChannels * height * ((width + 3) / 4) : outputCount
    let threadsPerGroup = 256
    let groups = MTLSize(width: (workItems + threadsPerGroup - 1) / threadsPerGroup,
                         height: 1, depth: 1)
    let threads = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
    let pipeline = pipelines[optimized ? "conv2d_optimized" : "conv2d_naive"]!
    let elapsed = timeKernel(warmup: item.warmup, repeats: item.repeats) {
        execute(queue) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(weights, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        }
    }
    let actual = outputValues(output, count: outputCount)
    var reference = [Float](repeating: 0, count: outputCount)
    for oc in 0..<outChannels {
        for oy in 0..<height {
            for ox in 0..<width {
                var sum: Float = 0
                for ic in 0..<inChannels {
                    for ky in 0..<3 {
                        for kx in 0..<3 {
                            let iy = oy + ky - 1, ix = ox + kx - 1
                            if iy < 0 || iy >= height || ix < 0 || ix >= width { continue }
                            let x = ic * height * width + iy * width + ix
                            let w = ((oc * inChannels + ic) * 3 + ky) * 3 + kx
                            sum += inputValues[x] * weightValues[w]
                        }
                    }
                }
                reference[oc * height * width + oy * width + ox] = sum
            }
        }
    }
    let error = maxError(actual, reference)
    let operations = Int64(2) * Int64(outputCount) * Int64(inChannels) * 9
    let bytes = Int64(inputValues.count + weightValues.count + outputCount) * 4
    let seconds = elapsed / 1_000.0
    return Measurement(operation: item.operation, size: height, device: "mps",
        precision: item.precision, variant: item.variant, median_ms: elapsed,
        operations: operations, gflops: Double(operations) / seconds / 1e9,
        modeled_bytes: bytes, estimated_gb_per_s: Double(bytes) / seconds / 1e9,
        arithmetic_intensity: Double(operations) / Double(bytes), max_abs_error: error,
        worker_threads: 0)
}

func runRowReductionCase(_ item: BenchmarkCase, device: MTLDevice, queue: MTLCommandQueue,
                         pipelines: [String: MTLComputePipelineState]) -> Measurement {
    let rows = item.operation == "softmax" ? 256 : 1024
    let columns = item.size
    let count = rows * columns
    let inputValues = vectorValues(count, seed: 17)
    let input = makeBuffer(device: device, values: inputValues)
    guard let output = device.makeBuffer(length: count * MemoryLayout<Float>.stride,
                                         options: .storageModeShared) else {
        fail("could not allocate row reduction output")
    }
    var columnCount = UInt32(columns)
    var rowCount = UInt32(rows)
    let optimized = item.variant == "optimized"
    let threads = MTLSize(width: 256, height: 1, depth: 1)
    let groups = optimized
        ? MTLSize(width: 1, height: rows, depth: 1)
        : MTLSize(width: (rows + 255) / 256, height: 1, depth: 1)
    let pipelineName = "\(item.operation)_\(item.variant)"
    guard let pipeline = pipelines[pipelineName] else { fail("missing pipeline \(pipelineName)") }
    let elapsed = timeKernel(warmup: item.warmup, repeats: item.repeats) {
        execute(queue) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&columnCount, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        }
    }
    let actual = outputValues(output, count: count)
    var reference = [Float](repeating: 0, count: count)
    for row in 0..<rows {
        let offset = row * columns
        if item.operation == "softmax" {
            let maxValue = (0..<columns).map { inputValues[offset + $0] }.max() ?? 0
            let exps = (0..<columns).map { exp(inputValues[offset + $0] - maxValue) }
            let denominator = exps.reduce(0, +)
            for col in 0..<columns { reference[offset + col] = exps[col] / denominator }
        } else {
            let mean = (0..<columns).reduce(Float(0)) { $0 + inputValues[offset + $1] } / Float(columns)
            let variance = (0..<columns).reduce(Float(0)) { sum, col in
                let delta = inputValues[offset + col] - mean
                return sum + delta * delta
            } / Float(columns)
            let invStd = 1 / sqrt(variance + 1e-5)
            for col in 0..<columns { reference[offset + col] = (inputValues[offset + col] - mean) * invStd }
        }
    }
    let error = maxError(actual, reference)
    let operations = Int64(5 * count)
    let transfers: Int64
    if item.operation == "softmax" { transfers = optimized ? 5 : 4 }
    else { transfers = optimized ? 3 : 4 }
    let bytes = transfers * Int64(count) * 4
    let seconds = elapsed / 1_000.0
    return Measurement(operation: item.operation, size: columns, device: "mps",
        precision: item.precision, variant: item.variant, median_ms: elapsed,
        operations: operations, gflops: Double(operations) / seconds / 1e9,
        modeled_bytes: bytes, estimated_gb_per_s: Double(bytes) / seconds / 1e9,
        arithmetic_intensity: Double(operations) / Double(bytes), max_abs_error: error,
        worker_threads: 0)
}

func runElementwiseCase(_ item: BenchmarkCase, device: MTLDevice, queue: MTLCommandQueue,
                        pipelines: [String: MTLComputePipelineState]) -> Measurement {
    let n = item.size
    let av = vectorValues(n, seed: 17)
    let bv = vectorValues(n, seed: 29)
    let a = makeBuffer(device: device, values: av)
    let b = makeBuffer(device: device, values: bv)
    let elementBytes = 4
    guard let output = device.makeBuffer(length: n * elementBytes,
                                         options: .storageModeShared) else {
        fail("could not allocate elementwise output")
    }
    var count = UInt32(n)
    let width = item.variant == "optimized" ? 4 : 1
    let logicalThreads = (n + width - 1) / width
    let threadsPerGroup = 256
    let groups = MTLSize(width: (logicalThreads + threadsPerGroup - 1) / threadsPerGroup,
                         height: 1, depth: 1)
    let threads = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
    let pipelineName = "elementwise_\(item.operation)_\(item.variant)"
    guard let pipeline = pipelines[pipelineName] else { fail("missing pipeline \(pipelineName)") }
    let elapsed = timeKernel(warmup: item.warmup, repeats: item.repeats) {
        execute(queue) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(a, offset: 0, index: 0)
            encoder.setBuffer(b, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        }
    }
    let actual = outputValues(output, count: n)
    let reference = zip(av, bv).map { item.operation == "add" ? $0 + $1 : $0 * $1 }
    let error = maxError(actual, reference)
    let operations = Int64(n)
    let bytes = Int64(3 * n * elementBytes)
    let seconds = elapsed / 1_000.0
    return Measurement(operation: item.operation, size: n, device: "mps",
        precision: item.precision, variant: item.variant, median_ms: elapsed,
        operations: operations, gflops: Double(operations) / seconds / 1e9,
        modeled_bytes: bytes, estimated_gb_per_s: Double(bytes) / seconds / 1e9,
        arithmetic_intensity: Double(operations) / Double(bytes), max_abs_error: error,
        worker_threads: 0)
}

func runFusedCase(_ item: BenchmarkCase, device: MTLDevice, queue: MTLCommandQueue,
                  pipelines: [String: MTLComputePipelineState]) -> Measurement {
    let n = item.size
    let av = vectorValues(n, seed: 17)
    let bv = vectorValues(n, seed: 29)
    let cv = vectorValues(n, seed: 43)
    let a = makeBuffer(device: device, values: av)
    let b = makeBuffer(device: device, values: bv)
    let c = makeBuffer(device: device, values: cv)
    let elementBytes = 4
    guard let output = device.makeBuffer(length: n * elementBytes, options: .storageModeShared),
          let temporary = device.makeBuffer(length: n * elementBytes, options: .storageModeShared) else {
        fail("could not allocate fused output buffers")
    }
    var count = UInt32(n)
    let threadsPerGroup = 256
    let groups = MTLSize(width: (n + threadsPerGroup - 1) / threadsPerGroup,
                         height: 1, depth: 1)
    let threads = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
    let elapsed = timeKernel(warmup: item.warmup, repeats: item.repeats) {
        if item.variant == "naive" {
            let multiply = pipelines["elementwise_multiply_naive"]!
            execute(queue) { encoder in
                encoder.setComputePipelineState(multiply)
                encoder.setBuffer(a, offset: 0, index: 0)
                encoder.setBuffer(b, offset: 0, index: 1)
                encoder.setBuffer(temporary, offset: 0, index: 2)
                encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
                encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
            }
            let add = pipelines["elementwise_add_naive"]!
            execute(queue) { encoder in
                encoder.setComputePipelineState(add)
                encoder.setBuffer(temporary, offset: 0, index: 0)
                encoder.setBuffer(c, offset: 0, index: 1)
                encoder.setBuffer(output, offset: 0, index: 2)
                encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
                encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
            }
        } else {
            let fused = pipelines["fused_mul_add"]!
            execute(queue) { encoder in
                encoder.setComputePipelineState(fused)
                encoder.setBuffer(a, offset: 0, index: 0)
                encoder.setBuffer(b, offset: 0, index: 1)
                encoder.setBuffer(c, offset: 0, index: 2)
                encoder.setBuffer(output, offset: 0, index: 3)
                encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 4)
                encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
            }
        }
    }
    let actual = outputValues(output, count: n)
    let reference = zip(zip(av, bv), cv).map { $0.0 * $0.1 + $1 }
    let error = maxError(actual, reference)
    let operations = Int64(2 * n)
    let transferCount = item.variant == "naive" ? 6 : 4
    let bytes = Int64(transferCount * n * elementBytes)
    let seconds = elapsed / 1_000.0
    return Measurement(operation: item.operation, size: n, device: "mps",
        precision: item.precision, variant: item.variant, median_ms: elapsed,
        operations: operations, gflops: Double(operations) / seconds / 1e9,
        modeled_bytes: bytes, estimated_gb_per_s: Double(bytes) / seconds / 1e9,
        arithmetic_intensity: Double(operations) / Double(bytes), max_abs_error: error,
        worker_threads: 0)
}

func main() throws {
    let input = FileHandle.standardInput.readDataToEndOfFile()
    let request = try JSONDecoder().decode(BenchmarkRequest.self, from: input)
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        fail("no Metal device or command queue found")
    }
    let sourcePath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent("kernels/metal/baseline.metal")
    let source = try String(contentsOf: sourcePath, encoding: .utf8)
    let library = try device.makeLibrary(source: source, options: nil)
    let names = [
        "matmul_naive", "matmul_tiled", "fused_mul_add",
        "elementwise_add_naive", "elementwise_add_optimized",
        "elementwise_multiply_naive", "elementwise_multiply_optimized",
        "conv2d_naive", "conv2d_optimized", "softmax_naive", "softmax_optimized",
        "layernorm_naive", "layernorm_optimized",
    ]
    var pipelines = [String: MTLComputePipelineState]()
    for name in names {
        guard let function = library.makeFunction(name: name) else { fail("missing function \(name)") }
        pipelines[name] = try device.makeComputePipelineState(function: function)
    }

    var results = [Measurement]()
    for item in request.cases {
        switch item.operation {
        case "matmul":
            results.append(runMatrixCase(item, device: device, queue: queue, pipelines: pipelines))
        case "fused_mul_add":
            results.append(runFusedCase(item, device: device, queue: queue, pipelines: pipelines))
        case "add", "multiply":
            results.append(runElementwiseCase(item, device: device, queue: queue, pipelines: pipelines))
        case "conv2d":
            results.append(runConvCase(item, device: device, queue: queue, pipelines: pipelines))
        case "softmax", "layernorm":
            results.append(runRowReductionCase(item, device: device, queue: queue,
                                               pipelines: pipelines))
        default:
            fail("unsupported custom Metal operation \(item.operation)")
        }
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    FileHandle.standardOutput.write(try encoder.encode(results))
}

do { try main() } catch { fail(error.localizedDescription) }
