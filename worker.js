async function withTimeout(promise, ms, label) {
  let timeoutId;

  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(`${label} timed out after ${ms}ms`));
    }, ms);
  });

  try {
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    clearTimeout(timeoutId);
  }
}

async function detectFacesFromUrl(imageUrl) {
  console.log(`[Worker] Loading image: ${imageUrl}`);

  const img = await withTimeout(loadImage(imageUrl), 20000, "loadImage");

  console.log(
    `[Worker] Image loaded: ${img.width}x${img.height}`
  );

  const detections = await withTimeout(
    faceapi
      .detectAllFaces(img)
      .withFaceLandmarks()
      .withFaceDescriptors(),
    45000,
    "face detection"
  );

  console.log(`[Worker] Detections found: ${detections.length}`);

  return detections.map((detection, index) => ({
    face_index: index,
    descriptor: Array.from(detection.descriptor),
    bounding_box: {
      x: detection.detection.box.x,
      y: detection.detection.box.y,
      width: detection.detection.box.width,
      height: detection.detection.box.height
    },
    confidence: detection.detection.score,
    face_width: Math.round(detection.detection.box.width),
    face_height: Math.round(detection.detection.box.height)
  }));
}

async function processPhoto(photo) {
  console.log(`[Worker] Processing photo ${photo.id}`);

  await markProcessing(photo.id);
  console.log(`[Worker] Marked processing ${photo.id}`);

  await clearExistingDescriptors(photo.id);
  console.log(`[Worker] Cleared old descriptors ${photo.id}`);

  const faces = await detectFacesFromUrl(photo.storage_url);

  await saveDescriptors(photo.event_id, photo.id, faces);
  console.log(`[Worker] Saved ${faces.length} descriptor row(s) for ${photo.id}`);

  await markComplete(photo.id, faces.length);
  console.log(`[Worker] Marked complete ${photo.id} with ${faces.length} face(s)`);
}
