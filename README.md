# Arabic Sign Language Translation App

A mobile application that translates Arabic Sign Language to text using React Native and Expo.

## Features

- Real-time sign language detection using device camera
- Clean and intuitive user interface
- Dark mode support
- Arabic language support

## Tech Stack

- React Native
- Expo
- TypeScript
- expo-camera for camera functionality
- React Navigation for routing

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Expo CLI
- Expo Go app on your mobile device

### Installation

1. Clone the repository
```bash
git clone https://github.com/AhmedOmani/Graduation-Project.git
cd Graduation-Project
```

2. Install dependencies
```bash
npm install
# or
yarn install
```

3. Start the development server
```bash
npx expo start
```

4. Scan the QR code with Expo Go (Android) or the Camera app (iOS)

## Project Structure

```
project/
├── app/
│   └── (tabs)/
│       ├── index.tsx      # Home screen
│       ├── camera.tsx     # Camera component
│       ├── translate.tsx  # Translation screen
│       └── history.tsx    # History screen
├── assets/
├── components/
├── hooks/
└── types/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors who have helped with the development
- Special thanks to the Expo team for their amazing framework
