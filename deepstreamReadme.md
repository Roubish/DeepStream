# DeepStream Inference Pipeline

This document describes the architecture of our DeepStream-based video analytics pipeline, the role of each plugin, and the rationale behind key design decisions (TensorRT, batching, tracking).

---

## Pipeline Overview

```
UriDecodeBin → Nvstreammux → Nvinfer (Primary/Detector) → Nvtracker → Nvinfer (Secondary/Classifier, optional) → Nvvideoconvert → Nvosd → Sink
```

`Queue` elements sit between adjacent plugins throughout the pipeline to decouple processing stages and enable asynchronous, multi-threaded execution.

```
[Source 1] ─┐
[Source 2] ─┼─ UriDecodeBin ─ Queue ─ Nvstreammux ─ Queue ─ Nvinfer ─ Queue ─ Nvtracker ─ Queue ─ Nvvideoconvert ─ Queue ─ Nvosd ─ Queue ─ Sink
[Source N] ─┘
```

---

## Plugin Reference

### 1. UriDecodeBin
Auto-detects the input source type — local file, RTSP stream, or USB/V4L2 camera — and automatically builds the correct demux + hardware-accelerated decode chain (NVDEC where available). Removes the need to manually assemble source-specific pipelines like `filesrc → qtdemux → h264parse → nvv4l2decoder`.

### 2. Nvstreammux
Batches frames from N independent, asynchronously-arriving streams into a single fixed-size batch buffer. Instantiates `NvDsBatchMeta`, the root metadata container that every downstream plugin writes to (`NvDsFrameMeta`, `NvDsObjectMeta`, etc.). This is the plugin that makes multi-stream pipelines possible — without it, nothing downstream is stream-aware.

**Key config params:** `batch-size`, `batched-push-timeout`, `width`, `height`.

### 3. Nvinfer
Wraps a TensorRT engine. Performs GPU-accelerated preprocessing (scale/normalize), runs inference on the batch, and parses raw output tensors into bounding boxes, classes, and confidence scores — attaching results as `NvDsObjectMeta`.

Runs in one of two modes:
- **Primary (PGIE):** operates on the full frame (e.g., object detector)
- **Secondary (SGIE):** operates on cropped objects found by the primary (e.g., classifier, attribute recognition)

### 4. Nvtracker
Sits immediately after the primary Nvinfer. Assigns a persistent `object_id` to detections across frames so the same object is tracked over time rather than being re-detected as "new" every frame.

Common tracker backends:
- **IOU** — lightweight, matches boxes by overlap
- **NvDCF** — correlation-filter / appearance-based, more robust to occlusion
- **NvSORT / DeepSORT-style** — appearance-embedding based re-identification

Enables downstream logic such as counting, line-crossing detection, dwell-time analysis, and re-identification.

### 5. Nvvideoconvert
Converts the raw NVMM (NVIDIA hardware memory) frame format/colorspace into the format required downstream (e.g., NV12 → RGBA). Also handles GPU-accelerated scaling, cropping, and rotation.

### 6. Nvosd
On-Screen Display: reads metadata (boxes, labels, tracker IDs, segmentation masks) and burns them onto the frame as an overlay using GPU-accelerated drawing.

### 7. Sink
Terminal pipeline element:
- `nveglglessink` / `nv3dsink` — on-screen display
- `filesink` (+ encoder) — save to disk
- `nvmsgbroker` / RTSP sink — stream out / send to a message broker

### Queue
A GStreamer element that decouples two adjacent plugins by giving each its own thread with an internal buffer. Prevents a slow element (e.g., inference) from stalling every stage upstream of it, so e.g. decoding for stream 3 can proceed while stream 1 is still mid-inference.

---

## Why TensorRT?

TensorRT is NVIDIA's inference optimizer and runtime. Given a trained model (ONNX, Caffe, UFF), it:

- **Fuses layers** (e.g., conv + bias + ReLU → one kernel) to reduce memory round-trips
- **Selects the fastest kernel implementation** per layer for the specific target GPU architecture
- **Quantizes to FP16/INT8** for higher throughput, using calibration to preserve accuracy
- **Produces a serialized `.engine` file** tuned to the exact GPU it was built on

Net effect: significantly lower latency and higher throughput than running the same model in a general-purpose framework — critical when running many concurrent streams in real time.

---

## Why Are Batches Required?

1. **GPU utilization.** GPUs are massively parallel throughput devices. Inferring on one frame at a time leaves most streaming multiprocessors idle. Batching N frames lets a single kernel launch process all of them, amortizing launch and memory-transfer overhead across the batch.

2. **Fixed compute graphs.** TensorRT engines are typically built for a fixed (or bounded dynamic) batch size. Feeding a consistent batch shape avoids re-planning per call and keeps memory allocation predictable.

3. **Multi-stream economics.** In production, inference isn't run on a single stream — it's run on 8/16/32+ concurrent RTSP cameras on one GPU. `Nvstreammux` collects one frame per stream into a single tensor, and `Nvinfer` processes the whole batch in one forward pass instead of running N separate inference calls competing for the GPU.

4. **Bandwidth-bound vs. compute-bound.** Small batches tend to be bandwidth-bound (time dominated by moving weights in/out of memory relative to compute performed). Larger batches shift the workload to compute-bound, where the GPU's raw FLOPs become the bottleneck — this is where the best throughput-per-watt and throughput-per-dollar is achieved.

### Trade-off: Throughput vs. Latency

Larger batches mean `Nvstreammux` waits longer to fill the batch before releasing it downstream. This is a tunable trade-off:

| Priority | Batch size | Config |
|---|---|---|
| Max throughput (offline/batch analytics) | Larger | Higher `batch-size`, higher `batched-push-timeout` |
| Low latency (live monitoring/alerting) | Smaller | Lower `batch-size`, lower `batched-push-timeout` |

---

## Summary Table

| Plugin | Role | GPU Accelerated |
|---|---|---|
| UriDecodeBin | Source detection + decode | Yes (NVDEC) |
| Nvstreammux | Batching + metadata init | Yes |
| Nvinfer | TensorRT inference | Yes |
| Nvtracker | Cross-frame object ID assignment | Yes |
| Nvvideoconvert | Format/colorspace conversion | Yes |
| Nvosd | Metadata overlay rendering | Yes |
| Sink | Display / save / stream out | Depends on sink type |
| Queue | Async decoupling between elements | N/A (CPU threading) |
