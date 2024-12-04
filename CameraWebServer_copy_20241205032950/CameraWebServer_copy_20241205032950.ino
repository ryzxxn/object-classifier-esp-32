#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include "img_converters.h" // Include the ESP32 library for image conversions
#include <ESP32Servo.h>     // Include the ESP32Servo library

// ===================
// Select camera model
// ===================
#define CAMERA_MODEL_AI_THINKER // Has PSRAM
#include "camera_pins.h"

// WiFi credentials
const char* ssid = "laptop";
const char* password = "12345678";

// Web server instance
WebServer server(80);

// Servo control pin
const int servoPin = 12; // Define the I/O pin for the servo
Servo servo;

// Function to handle image capture and convert to JPEG
void handleCapture() {
  camera_fb_t *fb = esp_camera_fb_get();  // Capture the image in RGB565
  if (!fb) {
    Serial.println("Camera capture failed");
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }

  if (fb->format != PIXFORMAT_RGB565) {
    Serial.println("Camera capture format is not RGB565!");
    server.send(500, "text/plain", "Camera capture format not RGB565");
    esp_camera_fb_return(fb); // Return frame buffer if format is incorrect
    return;
  }

  // Prepare a buffer to hold the JPEG data
  size_t jpeg_buf_len = 0;
  uint8_t *jpeg_buf = NULL;

  // Convert the RGB565 frame to JPEG format
  bool converted = frame2jpg(fb, 80, &jpeg_buf, &jpeg_buf_len); // 80 is the quality

  if (!converted) {
    Serial.println("JPEG conversion failed!");
    server.send(500, "text/plain", "JPEG conversion failed");
    esp_camera_fb_return(fb); // Return frame buffer if conversion fails
    return;
  }

  // Set the headers to force download
  server.sendHeader("Content-Disposition", "attachment; filename=capture.jpg"); // Forces download
  server.send_P(200, "image/jpeg", (const char*)jpeg_buf, jpeg_buf_len);         // Send the JPEG data

  // Free the JPEG buffer after sending the response
  free(jpeg_buf);

  // Return the original frame buffer to the camera driver
  esp_camera_fb_return(fb);
}

// Function to handle servo control
void handleServoControl() {
  if (server.hasArg("servo")) {
    int servoDegree = server.arg("servo").toInt();
    servo.write(servoDegree); // Move the servo to the specified degree
    delay(5000); // Wait for 5 seconds
    servo.write(0); // Reset the servo to 0 degrees
    server.send(200, "text/plain", "Servo moved to " + String(servoDegree) + " degrees and then reset to 0 degrees");
  } else {
    server.send(400, "text/plain", "Missing servo parameter");
  }
}

// Setup function to initialize camera and start the web server
void setup() {
  Serial.begin(115200);
  Serial.println();

  // Initialize servo control pin
  servo.attach(servoPin);

  // Camera configuration
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565; // Capture in RGB565 format
  config.frame_size = FRAMESIZE_VGA;     // Set frame size to QVGA (320x240)
  config.jpeg_quality = 12;               // JPEG quality (if converting)
  config.fb_count = 1;                    // Single frame buffer

  // Initialize the camera
  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed");
    while (1);  // Halt if camera initialization fails
  }

  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  // Define the endpoint for image capture and download
  server.on("/capture", HTTP_GET, handleCapture);

  // Define the endpoint for servo control
  server.on("/servo", HTTP_GET, handleServoControl);

  // Start the web server
  server.begin();
  Serial.println("Server started");
  Serial.print("Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("/capture' to capture and download image");
  Serial.print("Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("/servo?servo=<degree>' to control servo");
}

void loop() {
  // Handle incoming requests
  server.handleClient();
}
