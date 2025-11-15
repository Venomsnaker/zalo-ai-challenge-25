## YOLOe Pipelines
#### Visual-based pipeline
- Apply YOLOe on the footage using each provided image as reference.
- The bounding box of the reference image is also labeled by YOLOe using the object name.
- Prioritize the top down view of the image.
#### Text-based pipeline
- Use a VLM to construct a prompt that describe the provided images.
- Apply YOLOe on the footage using this prompt.
## Contributors
- Venomsnaker