import { useEffect } from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity, useColorScheme } from 'react-native';
import { useFonts, Cairo_700Bold, Cairo_400Regular } from '@expo-google-fonts/cairo';
import { useRouter } from 'expo-router';
import { Camera } from 'lucide-react-native';
import * as SplashScreen from 'expo-splash-screen';

SplashScreen.preventAutoHideAsync();

const DEFAULT_PROFILE_IMAGE = 'https://images.unsplash.com/photo-1633332755192-727a05c4013d?w=400&auto=format&fit=crop&q=80';

export default function HomeScreen() {
  const router = useRouter();
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  const [fontsLoaded] = useFonts({
    'Cairo-Bold': Cairo_700Bold,
    'Cairo-Regular': Cairo_400Regular,
  });

  useEffect(() => {
    if (fontsLoaded) {
      SplashScreen.hideAsync();
    }
  }, [fontsLoaded]);

  if (!fontsLoaded) {
    return null;
  }

  return (
    <View style={[styles.container, { backgroundColor: isDark ? '#111827' : '#f3f4f6' }]}>
      <View style={styles.header}>
        <Image
          source={{ uri: DEFAULT_PROFILE_IMAGE }}
          style={styles.profileImage}
        />
        <Text style={[styles.welcomeText, { color: isDark ? '#ffffff' : '#1f2937' }]}>
          مرحباً بك
        </Text>
      </View>

      <View style={styles.content}>
        <TouchableOpacity
          style={[styles.translateButton, { backgroundColor: '#2563eb' }]}
          onPress={() => router.push('/translate')}
        >
          <Camera size={48} color="#ffffff" />
          <Text style={styles.buttonText}>بدء الترجمة</Text>
        </TouchableOpacity>

        <Text style={[styles.description, { color: isDark ? '#9ca3af' : '#4b5563' }]}>
          اضغط على الزر للبدء في ترجمة لغة الإشارة العربية
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    padding: 20,
    flexDirection: 'row',
    alignItems: 'center',
  },
  profileImage: {
    width: 50,
    height: 50,
    borderRadius: 25,
  },
  welcomeText: {
    marginLeft: 15,
    fontSize: 24,
    fontFamily: 'Cairo-Bold',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  translateButton: {
    width: 200,
    height: 200,
    borderRadius: 100,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 4.65,
    elevation: 8,
  },
  buttonText: {
    color: '#ffffff',
    marginTop: 10,
    fontSize: 18,
    fontFamily: 'Cairo-Bold',
  },
  description: {
    marginTop: 30,
    textAlign: 'center',
    fontSize: 16,
    fontFamily: 'Cairo-Regular',
    paddingHorizontal: 40,
  },
});