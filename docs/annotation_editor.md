# Annotation Editor MVP

The v0.12 editor is a Public Alpha for **YOLO object-detection bounding boxes only**. It does not support segmentation, polygons, masks, pose/keypoints, OBB, tracking, video annotation, or AI/model-assisted annotation.

## Open a dataset

1. Open **Annotation Editor** in Simple or Advanced Mode.
2. Choose the dataset `data.yaml` and select **Open Dataset**.
3. Choose an available train, validation, or test split.
4. Select a class from the right panel. Class IDs and names come from the YAML `names` field.

The editor shares Dataset Check's YAML path resolution, including relative and absolute paths and the existing Roboflow-style compatibility rule. Nested `images/...` paths map to matching nested `labels/...` paths. Image formats are JPG/JPEG, PNG, BMP, and WebP. Duplicate image names that would map to the same label path are rejected rather than overwritten.

## Edit boxes

- **Draw Box:** drag inside the image to create a box using the active class.
- **Select:** click a box to select it, drag to move it, or drag one of its four corner handles to resize it.
- **Pan:** drag the view; middle-mouse drag also pans.
- The mouse wheel zooms from 10% through 800%; **Fit Image** and **100%** reset the view.
- Delete removes the selected box. Copy and paste duplicate normalized box geometry with a small clamped offset.
- Selecting another class changes the selected box class. Number keys 1–9 choose class IDs 0–8.

All scene geometry uses actual image pixels. The domain model stores normalized YOLO coordinates and converts through one shared `xyxy` conversion path.

## Save, autosave, and navigation

The output format is:

```text
class_id x_center y_center width height
```

Coordinates are written with a fixed six-decimal precision using a flushed temporary file followed by atomic replacement. Ctrl+S saves; Ctrl+Z/Ctrl+Y undo and redo within the current image. A/Left and D/Right move between images unless focus is in a text/editable field.

Autosave is enabled by default and runs only when the current document is dirty. Opening an unlabeled image and moving on does **not** create a label. If the last existing box is deleted and saved, the retained empty `.txt` is an intentional negative annotation. With Autosave disabled, navigation, dataset changes, split changes, and application close offer Save, Discard, and Cancel for dirty work.

## Malformed labels and backups

Wrong field counts, non-numeric values, invalid classes, NaN/Inf, zero-sized boxes, and out-of-range geometry are reported without crashing. Valid boxes in a partially malformed file remain visible, but normal save cannot silently discard invalid lines. **Repair and Save** requires explicit confirmation and first copies the original text to:

```text
%LOCALAPPDATA%\YOLO-Trainer-UI\annotation_backups\<path-hash>\<timestamp>-<label-name>
```

Ordinary valid-label saves do not create backups.

## Keyboard shortcuts

| Shortcut | Action |
|---|---|
| A / Left | Previous image |
| D / Right | Next image |
| Ctrl+S | Save |
| Ctrl+Z / Ctrl+Y | Undo / Redo |
| Delete | Delete selected box |
| Ctrl+C / Ctrl+V | Copy / Paste box |
| 1–9 | Choose class 0–8 |

## Alpha limitations

- Undo/redo history is scoped to the current image and clears on navigation.
- Unsaved in-memory work does not have a crash-recovery journal.
- The image list deliberately shows filenames and status rather than loading thousands of thumbnails.
- This release does not provide multi-select or eight-handle resizing.
- Public Alpha users should keep normal dataset backups and review labels before training.
