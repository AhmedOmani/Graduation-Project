import { View, Text, StyleSheet, FlatList, useColorScheme } from 'react-native';

// Example history data - in a real app, this would come from storage or an API
const historyData = [
  { id: '1', text: 'مرحباً', timestamp: '10:30 AM' },
  { id: '2', text: 'كيف حالك', timestamp: '11:45 AM' },
  { id: '3', text: 'شكراً', timestamp: '2:15 PM' },
];

export default function HistoryScreen() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  const renderItem = ({ item }: { item: typeof historyData[0] }) => (
    <View style={[styles.historyItem, { backgroundColor: isDark ? '#1f2937' : '#ffffff' }]}>
      <Text style={[styles.historyText, { color: isDark ? '#ffffff' : '#000000' }]}>
        {item.text}
      </Text>
      <Text style={[styles.timestamp, { color: isDark ? '#9ca3af' : '#6b7280' }]}>
        {item.timestamp}
      </Text>
    </View>
  );

  return (
    <View style={[styles.container, { backgroundColor: isDark ? '#111827' : '#f3f4f6' }]}>
      <FlatList
        data={historyData}
        renderItem={renderItem}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContainer}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  listContainer: {
    padding: 16,
  },
  historyItem: {
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  historyText: {
    fontSize: 18,
    fontFamily: 'Cairo-Bold',
    marginBottom: 8,
  },
  timestamp: {
    fontSize: 14,
    fontFamily: 'Cairo-Regular',
  },
});