import { useState, useRef, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, useColorScheme, ActivityIndicator } from 'react-native';
import { Camera, CameraType, CameraView, useCameraPermissions } from 'expo-camera';
import { Play, CircleStop as StopCircle } from 'lucide-react-native';

export default function CameraScreen() {
  const [permission, setPermission] = useCameraPermissions();
  const [isRecording, setIsRecording] = useState(false);
  const [translation, setTranslation] = useState<string | null>(null);
  const [type, setType] = useState<CameraType>("front");
  const cameraRef = useRef<typeof Camera | null>(null);
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  const toggleRecording = () => {
    if (isRecording) {
      setIsRecording(false);
      setTranslation(null);
    } else {
      setIsRecording(true);
      // Simulate translation (this will be replaced with actual model integration later)
      setTranslation('Recording...');
    }
  };

  if (!permission?.granted) {
    return (
      <View style={styles.container}>
        <Text style={[styles.text, isDark && styles.textDark]}>No access to camera</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        type={type}
        focusDepth={1}
      >
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={[styles.button, isDark && styles.buttonDark]}
            onPress={toggleRecording}
          >
            {isRecording ? (
              <StopCircle size={24} color={isDark ? '#fff' : '#000'} />
            ) : (
              <Play size={24} color={isDark ? '#fff' : '#000'} />
            )}
          </TouchableOpacity>
        </View>
      </CameraView>
      {translation && (
        <View style={styles.translationContainer}>
          <Text style={[styles.translationText, isDark && styles.textDark]}>
            {translation}
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    backgroundColor: 'transparent',
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'flex-end',
    marginBottom: 20,
  },
  button: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 50,
    marginHorizontal: 10,
  },
  buttonDark: {
    backgroundColor: '#333',
  },
  text: {
    fontSize: 18,
    color: '#000',
    textAlign: 'center',
  },
  textDark: {
    color: '#fff',
  },
  translationContainer: {
    position: 'absolute',
    bottom: 100,
    left: 0,
    right: 0,
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 10,
  },
  translationText: {
    fontSize: 24,
    color: '#fff',
    fontWeight: 'bold',
  },
});