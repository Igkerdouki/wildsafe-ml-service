# WildSafe ML Roadmap

## Completed

### Wildlife Detection
- [x] CLIP ViT-L/14 zero-shot classification (12 species)
- [x] Prompt engineering for high accuracy (95-99%)
- [x] Species: deer, elk, moose, bear, coyote, fox, raccoon, opossum, skunk, goat, horse, wild boar

### Human Safety Monitoring
- [x] Person detection (normal vs abnormal behavior)
- [x] MediaPipe pose estimation integration
- [x] Fallen person detection (body angle > 60°)
- [x] Distress recognition (hunched posture)

### API & Integration
- [x] FastAPI REST service
- [x] Image upload, base64, and URL endpoints
- [x] Video file processing
- [x] Speaker frequency output for hardware deterrence
- [x] Alert triggering (confidence > 70%)

### Hardware Integration
- [x] RPi4B speaker test (2kHz, 3kHz, 20kHz)
- [x] LED alert system prototype
- [x] Speaker frequency mapping per species

## In Progress
- [ ] Azure cloud deployment
- [ ] Real-time camera stream integration
- [ ] RPi4B full integration with ML API

## Next Steps
- [ ] WebSocket streaming endpoint
- [ ] Night-time optimization (IR camera support)
- [ ] Incident recording and logging
- [ ] Sensor fusion (motion/proximity sensors)

## Future Expansion
- [ ] Google Maps driver alerts
- [ ] Predictive collision risk zones
- [ ] Analytics dashboard
- [ ] Multi-camera support
- [ ] Edge deployment (TensorRT/ONNX optimization)
