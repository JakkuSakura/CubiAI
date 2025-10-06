# GUI Roadmap

The GUI concept shifts from segmentation review toward motion-transfer experimentation. The goal is a lightweight PySide6 application that helps visualise motion descriptors, compare generated frames, and manage experiments without leaving the desktop.

## Milestones
1. **Prototype Viewer**
   - Load a portrait, driver clip, and generated outputs side-by-side.
   - Scrub through timelines with overlayed keypoints or descriptor heatmaps.
   - Display training metrics for the selected checkpoint.

2. **Experiment Manager**
   - Launch training jobs with preset configs, monitor progress, and archive results.
   - Tag runs with descriptor variants, loss weights, and dataset subsets for easy comparison.

3. **Descriptor Inspector**
   - Visualise discovered keypoints, attention maps, or canonical coordinates directly on frames.
   - Provide before/after toggles to spot colour drift or structural artefacts.

4. **Batch Preview & Export**
   - Queue multiple inference runs (different portraits or drivers) and export GIF/MP4 previews.
   - Generate diagnostic reports summarising motion alignment scores once the evaluation suite lands.

## Technical Notes
- **PySide6/Qt** remains the UI toolkit.
- Worker threads (QtConcurrent/QThreadPool) will run inference or descriptor extraction asynchronously.
- Rendering overlays may use Qt’s `QPainter` or a small OpenGL canvas for performance.
- IPC hooks (e.g., ZeroMQ or WebSockets) can stream metrics from long-running training processes.

## Design Principles
- Keep the GUI optional; the CLI should remain the single source of truth.
- Optimise for quick qualitative review—artists should preview motion transfers in seconds.
- Make experiment provenance explicit: show commit, configuration hash, and dataset fingerprint.

This roadmap will adjust as the descriptor stack matures and we gather feedback on which visualisations are most useful for diagnosing motion transfer quality.
