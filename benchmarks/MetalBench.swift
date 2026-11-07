import Foundation
import Metal
import Dispatch

struct BenchmarkCase: Decodable {
    let operation: String
    let size: Int
    let variant: String
    let warmup: Int
    let repeats: Int
}

struct BenchmarkRequest: Decodable {
    let cases: [BenchmarkCase]
}

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

func runMatrixCase(_ item: BenchmarkCase, device: MTLDevice,
                   queue: MTLCommandQueue, pipelines: [String: MTLComputePipelineState],
                   referenceBySize: inout [Int: [Float]]) -> Measurement {
    let n = item.size
    let count = n * n
    let a = makeBuffer(device: device, values: matrixValues(n, seed: 17))
    let b = makeBuffer(device: device, values: matrixValues(n, seed: 29))
    guard let output = device.makeBuffer(length: count * MemoryLayout<Float>.stride,
                                        options: .storageModeShared) else {
        fail("could not allocate GEMM output")
    }
    let pipelineName = item.variant == "tiled" ? "matmul_tiled" : "matmul_naive"
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
    let values = outputValues(output, count: count)
    let error: Double
    if item.variant == "naive" {
        referenceBySize[n] = values
        error = 0.0
    } else if let reference = referenceBySize[n] {
        error = maxError(values, reference)
    } else {
        fail("tiled GEMM requires a naive GEMM reference for size \(n)")
    }

    let operations = Int64(2) * Int64(n) * Int64(n) * Int64(n)
    let elementsRead: Int64
    if item.variant == "naive" {
        elementsRead = 2 * Int64(n) * Int64(n) * Int64(n)
    } else {
        let tiles = Int64((n + 15) / 16)
        elementsRead = 2 * Int64(n) * Int64(n) * tiles
    }
    let bytes = (elementsRead + Int64(count)) * Int64(MemoryLayout<Float>.stride)
    let seconds = elapsed / 1_000.0
    return Measurement(
        operation: "matmul", size: n, device: "mps", precision: "fp32",
        variant: item.variant, median_ms: elapsed, operations: operations,
        gflops: Double(operations) / seconds / 1e9, modeled_bytes: bytes,
        estimated_gb_per_s: Double(bytes) / seconds / 1e9,
        arithmetic_intensity: Double(operations) / Double(bytes), max_abs_error: error
    )
}

func runFusedCase(_ item: BenchmarkCase, device: MTLDevice,
                  queue: MTLCommandQueue, pipeline: MTLComputePipelineState) -> Measurement {
    let n = item.size
    let av = vectorValues(n, seed: 17)
    let bv = vectorValues(n, seed: 29)
    let cv = vectorValues(n, seed: 43)
    let a = makeBuffer(device: device, values: av)
    let b = makeBuffer(device: device, values: bv)
    let c = makeBuffer(device: device, values: cv)
    guard let output = device.makeBuffer(length: n * MemoryLayout<Float>.stride,
                                         options: .storageModeShared) else {
        fail("could not allocate fused output")
    }
    var count = UInt32(n)
    let threadsPerGroup = 256
    let groups = MTLSize(width: (n + threadsPerGroup - 1) / threadsPerGroup,
                         height: 1, depth: 1)
    let threads = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
    let encode = { (encoder: MTLComputeCommandEncoder) in
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(b, offset: 0, index: 1)
        encoder.setBuffer(c, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
    }
    let elapsed = timeKernel(warmup: item.warmup, repeats: item.repeats) {
        execute(queue, encode)
    }
    let values = outputValues(output, count: n)
    let reference = zip(zip(av, bv), cv).map { $0.0 * $0.1 + $1 }
    let error = maxError(values, reference)
    let operations = Int64(2 * n)
    // One read of each input and one output write for the fused pass.
    let bytes = Int64(4 * n * MemoryLayout<Float>.stride)
    let seconds = elapsed / 1_000.0
    return Measurement(
        operation: "fused_mul_add", size: n, device: "mps", precision: "fp32",
        variant: "metal_fused", median_ms: elapsed, operations: operations,
        gflops: Double(operations) / seconds / 1e9, modeled_bytes: bytes,
        estimated_gb_per_s: Double(bytes) / seconds / 1e9,
        arithmetic_intensity: Double(operations) / Double(bytes), max_abs_error: error
    )
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
    var pipelines = [String: MTLComputePipelineState]()
    for name in ["matmul_naive", "matmul_tiled", "fused_mul_add"] {
        guard let function = library.makeFunction(name: name) else { fail("missing function \(name)") }
        pipelines[name] = try device.makeComputePipelineState(function: function)
    }

    var referenceBySize = [Int: [Float]]()
    var results = [Measurement]()
    for item in request.cases {
        switch item.operation {
        case "matmul":
            results.append(runMatrixCase(item, device: device, queue: queue,
                                          pipelines: pipelines, referenceBySize: &referenceBySize))
        case "fused_mul_add":
            results.append(runFusedCase(item, device: device, queue: queue,
                                        pipeline: pipelines["fused_mul_add"]!))
        default:
            fail("unsupported custom Metal operation \(item.operation)")
        }
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    FileHandle.standardOutput.write(try encoder.encode(results))
}

do {
    try main()
} catch {
    fail(error.localizedDescription)
}
