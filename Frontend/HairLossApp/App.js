import React, { useState } from 'react';
import { StyleSheet, Text, View, Button, Image, ActivityIndicator, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const API_URL = "http://192.168.137.1:8000/predict/";

export default function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setResult(null); // Clear previous result
    }
  };

  const uploadImage = async () => {
    if (!image) {
      Alert.alert("Error", "Please select an image first");
      return;
    }

    setLoading(true);

    let formData = new FormData();
    let filename = image.split('/').pop();
    let match = /\.(\w+)$/.exec(filename);
    let type = match ? `image/${match[1]}` : `image`;

    formData.append('file', {
      uri: image,
      name: filename,
      type: type
    });

    try {
      let response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      let json = await response.json();
      setResult(json);
    } catch (error) {
      console.error(error);
      Alert.alert("Error", "Failed to connect to the server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Hair Loss Stage Classifier</Text>

      <Button title="Pick an image from camera roll" onPress={pickImage} />

      {image && <Image source={{ uri: image }} style={styles.image} />}

      <View style={styles.buttonContainer}>
        <Button title="Analyze Image" onPress={uploadImage} disabled={!image || loading} />
      </View>

      {loading && <ActivityIndicator size="large" color="#0000ff" />}

      {result && (
        <View style={styles.resultContainer}>
          <Text style={styles.resultText}>Stage: {result.predicted_stage}</Text>
          <Text style={styles.resultText}>Confidence: {result.confidence}</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 20 },
  title: { fontSize: 20, fontWeight: 'bold', marginBottom: 20 },
  image: { width: 224, height: 224, marginVertical: 20, borderRadius: 10 },
  buttonContainer: { marginVertical: 10 },
  resultContainer: { marginTop: 20, padding: 15, backgroundColor: '#f0f0f0', borderRadius: 10 },
  resultText: { fontSize: 16, fontWeight: 'bold', marginVertical: 5 }
});