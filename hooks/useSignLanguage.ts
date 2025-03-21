import { useState, useCallback } from 'react';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import { CameraCapturedPicture, CameraType } from 'expo-camera';

const API_URL = 'http://localhost:5000/predict';

type CameraInstance = {
  takePictureAsync: (options?: {
    quality?: number;
    base64?: boolean;
    skipProcessing?: boolean;
  }) => Promise<CameraCapturedPicture>;
};

export const useSignLanguage = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);

  const processFrame = useCallback(async (camera: CameraInstance | null) => {
    if (!camera || isProcessing) return;

    try {
      setIsProcessing(true);

      // Take a photo
      const photo = await camera.takePictureAsync({
        quality: 0.5,
        base64: true,
        skipProcessing: true,
      });

      // Compress and resize the image
      const processedImage = await manipulateAsync(
        photo.uri,
        [{ resize: { width: 640 } }],
        { compress: 0.7, format: SaveFormat.JPEG, base64: true }
      );

      if (!processedImage.base64) {
        throw new Error('Failed to process image');
      }

      // Send to Python backend
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: processedImage.base64,
        }),
      });

      const result = await response.json();

      if (result.success && result.prediction) {
        setPrediction(result.prediction);
      }
    } catch (error) {
      console.error('Error processing frame:', error);
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing]);

  return {
    isProcessing,
    prediction,
    processFrame,
  };
};
