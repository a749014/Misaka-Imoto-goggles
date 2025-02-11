#include <WiFi.h>

// 定义热点的SSID和密码
const char* ssid = "MisakaNetwork";
const char* password = "ZXC741ASD852QWE963";


void setup() {
  Serial.begin(9600);

  // 创建热点
  WiFi.softAP(ssid, password);
  //  打印热点 IP
  Serial.print("Wi-Fi 接入的 IP：");
  Serial.println(WiFi.softAPIP());

}

void loop() {

}
