# GUI Roadmap (Phase Two)

The PySide GUI will build upon the CLI core to deliver an artist-friendly experience. This roadmap describes planned features, UX milestones, and technical considerations.

## Milestones

1. **Foundational GUI Shell**
   - Project launcher with configuration selection and recent workspaces.
   - Embedded console view mirroring CLI logs.
   - Background worker infrastructure for long-running tasks.

2. **Segmentation Review Tools**
   - Layer preview canvas with toggles for masks and RGBA outputs.
   - Brush-based touch-up tools that feed updated masks back into the pipeline.
   - Confidence heatmaps overlay for quick QA.

3. **Rig Visualization**
   - Hierarchical tree for parts and deformers.
   - Parameter curve widgets for tweaking idle/breathing motions.
   - Live preview panel using Cubism SDK for Python or Qt WebEngine integration with the Cubism Viewer.

4. **Export & Validation Wizards**
   - Guided workflow to review diagnostics, adjust export settings, and bundle outputs.
   - Validation checklist ensuring Cubism import readiness.

5. **Quality-of-Life Enhancements**
   - Undo/redo stack for manual edits.
   - Asset library for commonly reused textures or rig templates.
   - Plugin marketplace integration (e.g., install new segmentation providers).

## Technical Stack
- **PySide6** for Qt-based UI components.
- **QtConcurrent/QThreadPool** for asynchronous processing.
- **qasync** bridge for integrating asyncio-based pipeline steps.
- **matplotlib / QtGraph** for plotting diagnostics.
- **OpenGL** surface or Cubism SDK viewer for real-time preview.

## Integration with CLI Core
- The GUI imports the same pipeline modules used by the CLI to avoid divergence.
- Workspace events are emitted through a message bus (e.g., `asyncio.Queue` or `pydantic` event models).
- Configuration edits in the GUI persist back to YAML configuration files per project.

## UX Principles
- Provide safe defaults while allowing experts to drill into underlying parameters.
- Surface model provenance and licensing within the UI when third-party assets are used.
- Support keyboard-driven workflows to accommodate power users.

## Testing Strategy
- Unit tests for view models and service layer.
- Screenshot tests for key dialogs using Qt test utilities.
- End-to-end smoke tests scripted with `pytest-qt`.

This roadmap will be refined after the CLI core stabilizes and we collect feedback from early adopters.
