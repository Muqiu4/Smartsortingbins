#include<AccelStepper.h>
#include<Servo.h>
#include <NewPing.h>
Servo fall_platform,spin_platform,leftservo,rightservo,clawservo,spinServo,dam_board_One,dam_board_Two,dam_board_Three,dam_board_Four;
char charData[20];
int intData[30];
int newintData[10];
int charIndex = 0;
int intIndex = 0;
int tempInt = 0;
int newintIndex = 0;
int rubbish_quality = 0;
bool isReadingInt = false;
const int fall_platform_Pin = 8;
const int spin_platform_Pin = 7;
const int platform_servospeed = 5;
const int platform_servospeed_Two = 10;
const int fall_platform_initial_position = 95;
const int spin_platform_initial_position = 23;
bool is_spin_big = false;
bool is_fall_big = false;
const int h_spin_platform_position=30;
const int h_fall_platform_position=0;
const int k_spin_platform_position=10;
const int k_fall_platform_position=180;
const int e_spin_platform_position=120;
const int e_fall_platform_position=0;
const int To_position = 70;
const int dam_board_One_Pin = 39;   
const int dam_board_Two_Pin = 37;   
const int dam_board_Three_Pin = 36; 
const int dam_board_Four_Pin = 40;  
const int dam_board_One_Maxpos = 105;
const int dam_board_One_Minpos = 40;
const int dam_board_Two_Maxpos = 100;
const int dam_board_Two_Minpos = 40;
const int dam_board_Three_Maxpos = 90;
const int dam_board_Three_Minpos = 40;
const int dam_board_Four_Maxpos = 110;
const int dam_board_Four_Minpos = 40;
const int dam_board_servospeed = 2;
const int dam_board_movePos = 50;
const int dam_board_shirking_movePos = 20;
const int XdirPin =  2;
const int XstepPin = 3;
const int YdirPin =  4;
const int YstepPin = 5;
AccelStepper Xstepper(1,XstepPin,XdirPin);
AccelStepper Ystepper(1,YstepPin,YdirPin);
const int r_Xcoordinates_value = 1350;
const int r_Ycoordinates_value = 150;
const int k_Xcoordinates_value = 1000;
const int k_Ycoordinates_value = 1500;
const int h_Xcoordinates_value = 50;
const int h_Ycoordinates_value = 50;
const int e_Xcoordinates_value =  200;
const int e_Ycoordinates_value = 1500;
int  X_move_steps = 0;
int  Y_move_steps = 0;
const int SW1 = 22;
const int SW2 = 23;
const int K1 =  24;
const int K2 =  25;
const int compress_time =800;
const int r_TRIGGER_Pin = 50;
const int r_ECHO_Pin =    51;
const int h_TRIGGER_Pin = 46;
const int h_ECHO_Pin =    47;
const int e_TRIGGER_Pin = 30;
const int e_ECHO_Pin =    31;
const int k_TRIGGER_Pin = 34;
const int k_ECHO_Pin =    35;
const int Measure_distance = 6;
unsigned long previousMillis = 0;
unsigned long currentMillis = 0;
const unsigned int MEASURE_INTERVAL = 500; 
NewPing r_sonar(r_TRIGGER_Pin, r_ECHO_Pin);
NewPing h_sonar(h_TRIGGER_Pin, h_ECHO_Pin);
NewPing e_sonar(e_TRIGGER_Pin, e_ECHO_Pin);
NewPing k_sonar(k_TRIGGER_Pin, k_ECHO_Pin);
const int left_arm_Pin = 9;
const int right_arm_Pin = 10;
const int claw_Pin = 11;
const int spinServo_Pin = 12;
const int left_arm_initial_position = 84; 
const int right_arm_initial_position = 96;
const int r_Arm_movePos = 30; 
const int e_Arm_movePos = 40;
const int claw_max_pos = 90;
const int claw_min_pos = 20; 
const int r_claw_Pos = 40;
const int h_claw_Pos = 20;
const int e1_claw_Pos = 35;
int e_claw_Pos = 0;
const int recycable_press_Pos = 80;
const int spin_initial_pos = 80;
const int spin_pos = 180;
bool  is_spin = false;
const int arm_servospeed = 3;
const int claw_servospeed = 8;
void setup() {
  Serial.begin(9600);
  fall_platform.attach(fall_platform_Pin);
  fall_platform.write(fall_platform_initial_position);
  spin_platform.attach(spin_platform_Pin);
  spin_platform.write(spin_platform_initial_position);
  dam_board_One.attach(dam_board_One_Pin);
  dam_board_One.write(dam_board_One_Maxpos);
  dam_board_Two.attach(dam_board_Two_Pin);
  dam_board_Two.write(dam_board_Two_Maxpos);
  dam_board_Three.attach(dam_board_Three_Pin);
  dam_board_Three.write(dam_board_Three_Maxpos);
  dam_board_Four.attach(dam_board_Four_Pin);
  dam_board_Four.write(dam_board_Four_Maxpos);
  pinMode(XstepPin,OUTPUT);
  pinMode(XdirPin,OUTPUT);
  pinMode(YstepPin,OUTPUT);
  pinMode(YdirPin,OUTPUT);
  Xstepper.setMaxSpeed(4000.0);     
  Xstepper.setAcceleration(2000.0); 
  Ystepper.setMaxSpeed(4000.0);     
  Ystepper.setAcceleration(2000.0); 
  Xstepper.setPinsInverted(true, false, false);
  Ystepper.setPinsInverted(true, false, false);
  pinMode(SW1,OUTPUT);
  pinMode(SW2,OUTPUT);
  pinMode(K1,OUTPUT);
  pinMode(K2,OUTPUT);
  digitalWrite(SW1,1);
  digitalWrite(SW2,1);
  digitalWrite(K1,1);
  digitalWrite(K2,1);
  pinMode(r_TRIGGER_Pin,OUTPUT);
  pinMode(r_ECHO_Pin,INPUT);
  pinMode(h_TRIGGER_Pin,OUTPUT);
  pinMode(h_ECHO_Pin,INPUT);
  pinMode(e_TRIGGER_Pin,OUTPUT);
  pinMode(e_ECHO_Pin,INPUT);
  pinMode(k_TRIGGER_Pin,OUTPUT);
  pinMode(k_ECHO_Pin,INPUT);
  leftservo.attach(left_arm_Pin);  
  leftservo.write(left_arm_initial_position);
  rightservo.attach(right_arm_Pin);  
  rightservo.write(right_arm_initial_position);
  clawservo.attach(claw_Pin);
  clawservo.write(claw_max_pos);
  spinServo.attach(spinServo_Pin);
  spinServo.write(spin_initial_pos);
}
void loop() {
  currentMillis = millis();
  detect_distance();
  while (Serial.available()) {
    char c = Serial.read();
    if (isDigit(c)) {
      tempInt = tempInt * 10 + (c - '0');
      isReadingInt = true;
    } else {
      if (isReadingInt) {
        intData[intIndex++] = tempInt;
        tempInt = 0;
        isReadingInt = false;
      }
      charData[charIndex++] = c;
    }
    if (isReadingInt) { 
      intData[intIndex++] = tempInt;
      tempInt = 0;
      isReadingInt = false;
    } 
  if(charData[charIndex-1] == 'o'){
    deal_digital();
    charData[charIndex-1] = 'l';
  }
}
}
void deal_digital(){
  rubbish_quality = intData[0];
  int j = 0;
  for( int i = 1;i < intIndex ;i = i+3){
    newintData[j] = intData[i]*100 + intData[i+1]*10 + intData[i+2];
    j++;
  }
  newintIndex = (intIndex - 1)/3;
  test_data();
}
void test_data(){
  int sum = 0;
  for(int i = 0;i < newintIndex - 1;i++){
    sum += newintData[i];
  }
  int ave = sum/(newintIndex - 1);
  if(ave == newintData[newintIndex - 1]){
    delay(10);
    Serial.print(1);
    if(rubbish_quality == 1){
      identify_rubbish_one();
    }else{
      identify_rubbish_two();
    }
    charIndex = 0;
    intIndex = 0;
    newintIndex = 0;
    delay(100);
    Serial.print(2);
  }else{
    delay(100);
    Serial.print(0);
    charIndex = 0;
    intIndex = 0;
    newintIndex = 0;
  }
}
void identify_rubbish_one(){
    char temp = 0;
    if(charData[0] == 'r'||charData[0] == 'h'||charData[0] == 'e'||charData[0] == 'k'){
      temp = charData[0];}else{
      temp = charData[1];
    }
    switch(temp){
    case 'r':{
        identify_rubbish_two();
        break;
      }
    case 'h':{
      delay(20);
      dam_board_shirking();
      delay(100);
      open_dam_board();
      delay(300);
      platform_dump(h_spin_platform_position,h_fall_platform_position);
      delay(300);
      close_dam_board();
      break;}
    case 'k':{
      delay(20);
      dam_board_shirking();
      delay(100);
      open_dam_board();
      delay(300);
      platform_dump(k_spin_platform_position,k_fall_platform_position);
      delay(300);
      close_dam_board();
      break;}
    case 'e':{
      delay(20);
      dam_board_shirking();
      delay(100);
      open_dam_board();
      delay(300);
      platform_dump(e_spin_platform_position,e_fall_platform_position);
      delay(300);
      close_dam_board();
      break;}
  }
}
void identify_rubbish_two(){
  char temp = 0;
  if(charData[0] == 'r'||charData[0] == 'h'||charData[0] == 'e'||charData[0] == 'k'){
    temp = charData[0];
  }else{
    temp = charData[1];}
  coordinates_computed();
  switch(temp){
    case 'r':{
      delay(20);
      open_dam_board();
      delay(100);
      move_Xstepper(X_move_steps);
      delay(500);
      move_Ystepper(Y_move_steps);
      delay(300);
      if(is_spin){
        spin_Arm();
      }
      delay(300);
        arm_down(r_Arm_movePos);
        delay(300);
        claw_clamp(r_claw_Pos);
        delay(600);
        arm_up(e_Arm_movePos);
      delay(200);
      if(is_spin){
        spin_Arm_back();
        is_spin = false;
      }
      delay(300);
      move_Xstepper(r_Xcoordinates_value);
      delay(500);
      move_Ystepper(r_Ycoordinates_value);
      delay(300);
      arm_down(e_Arm_movePos);
      delay(600);
      claw_release(e_claw_Pos);
      delay(200);
      arm_up(e_Arm_movePos);
      push_rod_move_forword();
      delay(200);
      stepMotor_Moveback();
      delay(100);
      close_dam_board();
      delay(3500);
      push_rod_move_back();
      break;
    }
  case 'h':{
      delay(20);
      dam_board_shirking();
      delay(150);
      open_dam_board();
      delay(100);
      move_Xstepper(X_move_steps);
      delay(500);
      move_Ystepper(Y_move_steps);
      delay(300);
      if(is_spin){
        spin_Arm();
      }
      delay(300);
        arm_down(e_Arm_movePos);
        delay(300);
        claw_clamp(e_claw_Pos);
        delay(600);
        arm_up(e_Arm_movePos);
      if(is_spin){
        spin_Arm_back();
        is_spin = false;
      }
      delay(300);
      move_Xstepper(h_Xcoordinates_value);
      delay(500);
      move_Ystepper(h_Ycoordinates_value);
      delay(300);
      arm_down(e_Arm_movePos);
      delay(600);
      claw_release(e_claw_Pos);
      delay(200);
      arm_up(e_Arm_movePos);
      stepMotor_Moveback();
      close_dam_board();
      break;
    }
  case 'k':{
      delay(20);
      dam_board_shirking();
      delay(150);
      open_dam_board();
      delay(100);
      move_Xstepper(X_move_steps);
      delay(500);
      move_Ystepper(Y_move_steps);
      delay(300);
      if(is_spin){
        spin_Arm();
      }
      delay(300);
        arm_down(e_Arm_movePos);
        delay(300);
        claw_clamp(e_claw_Pos);
        delay(600);
        arm_up(e_Arm_movePos);
      if(is_spin){
        spin_Arm_back();
        is_spin = false;
      }
      delay(300);
      move_Xstepper(k_Xcoordinates_value);
      delay(500);
      move_Ystepper(k_Ycoordinates_value);
      delay(300);
      arm_down(e_Arm_movePos);
      delay(600);
      claw_release(e_claw_Pos);
      delay(200);
      arm_up(e_Arm_movePos);
      stepMotor_Moveback();
      close_dam_board();
      break;
    }
  case 'e':{
      delay(20);
      dam_board_shirking();
      delay(150);
      open_dam_board();
      delay(100);
      move_Xstepper(X_move_steps);
      delay(500);
      move_Ystepper(Y_move_steps);
      delay(300);
      if(is_spin){
        spin_Arm();
      }
      delay(300);
        arm_down(e_Arm_movePos);
        delay(300);
        claw_clamp(e_claw_Pos);
        delay(600);
        arm_up(e_Arm_movePos);
      if(is_spin){
        spin_Arm_back();
        is_spin = false;
      }
      delay(300);
      move_Xstepper(e_Xcoordinates_value);
      delay(500);
      move_Ystepper(e_Ycoordinates_value);
      delay(300);
      arm_down(e_Arm_movePos);
      delay(600);
      claw_release(e_claw_Pos);
      delay(200);
      arm_up(e_Arm_movePos);
      stepMotor_Moveback();
      close_dam_board();
      break;
    }
  }
}
void  platform_dump(int spin_platform_position,int fall_platform_position){//20,85
  if(spin_platform_initial_position > spin_platform_position){
    for(int i = spin_platform_initial_position; i >= spin_platform_position; i--){
      spin_platform.write(i);
      delay(platform_servospeed);}
      is_spin_big = true;
    }else if(spin_platform_initial_position == spin_platform_position){
      spin_platform.write(spin_platform_initial_position);
    }else{
      for(int i = spin_platform_initial_position; i <= spin_platform_position; i++){
      spin_platform.write(i);
      delay(platform_servospeed);}
    }
  delay(100);
  if(fall_platform_initial_position > fall_platform_position){
      for(int i = fall_platform_initial_position; i >= fall_platform_position; i--){
      fall_platform.write(i);
      delay(platform_servospeed);}
      is_fall_big = true;
    }else if(fall_platform_initial_position == fall_platform_position){
        fall_platform.write(fall_platform_initial_position);
    }else{
      for(int i = fall_platform_initial_position; i <= fall_platform_position; i++){
      fall_platform.write(i);
      delay(platform_servospeed);}
  }
  delay(100);
  if(is_fall_big){
      for(int i = fall_platform_position; i <= fall_platform_initial_position; i++){
      fall_platform.write(i);
      delay(platform_servospeed);}
      is_fall_big = false;
    }else{
      for(int i = fall_platform_position; i >= fall_platform_initial_position; i--){
      fall_platform.write(i);
      delay(platform_servospeed);}
  }
    delay(100);
  if(is_spin_big){
    for(int i = spin_platform_position; i <= spin_platform_initial_position; i++){
    spin_platform.write(i);
    delay(platform_servospeed);}
    is_spin_big = false;
  }else{
    for(int i = spin_platform_position; i >= spin_platform_initial_position; i--){
    spin_platform.write(i);
    delay(platform_servospeed);}
  }
}
void dam_board_shirking()
{
  for(int i = 0;i <= dam_board_shirking_movePos;i++)
  {
      dam_board_One.write(dam_board_One_Maxpos + i);
      delay(dam_board_servospeed);
  }
  delay(200);
  for(int i = 0;i <= dam_board_shirking_movePos;i++)
  {
      dam_board_Two.write(dam_board_Two_Maxpos + i);
      delay(dam_board_servospeed);
  }
    delay(200);
    for(int i = 0;i <= dam_board_shirking_movePos;i++)
  {
      dam_board_Three.write(dam_board_Three_Maxpos + i);
      delay(dam_board_servospeed);
  }
   delay(200);
  for(int i = 0;i <= dam_board_shirking_movePos;i++)
  {
      dam_board_Four.write(dam_board_Four_Maxpos + i);
      delay(dam_board_servospeed);
  }
  delay(200);
    for(int i = 0;i <= dam_board_shirking_movePos;i++)
  {
      dam_board_Four.write(dam_board_Four_Maxpos + dam_board_shirking_movePos - i);
      dam_board_Three.write(dam_board_Three_Maxpos + dam_board_shirking_movePos - i);
      dam_board_Two.write(dam_board_Two_Maxpos + dam_board_shirking_movePos - i);
      dam_board_One.write(dam_board_One_Maxpos + dam_board_shirking_movePos - i);
      delay(dam_board_servospeed);
  }
}
void open_dam_board(){
  for(int i = 0; i <= dam_board_movePos; i++){
    dam_board_One.write(dam_board_One_Maxpos - i);
    dam_board_Two.write(dam_board_Two_Maxpos - i);
    dam_board_Three.write(dam_board_Three_Maxpos - i);
    dam_board_Four.write(dam_board_Four_Maxpos - i);
  delay(dam_board_servospeed);}
}
void close_dam_board(){//40--100
  for(int i = 0; i <= dam_board_movePos; i++){
    dam_board_One.write(dam_board_One_Maxpos - dam_board_movePos + i);
    dam_board_Two.write(dam_board_Two_Maxpos - dam_board_movePos + i);
    dam_board_Three.write(dam_board_Three_Maxpos - dam_board_movePos + i);
    dam_board_Four.write(dam_board_Four_Maxpos - dam_board_movePos + i);
  delay(dam_board_servospeed);}
}
void coordinates_computed(){
  int  X_coordinate_value =  0.45834*((newintData[0]+newintData[2])/2) - 30;
  int  Y_coordinate_value =  0.45834*((newintData[1]+newintData[3])/2);
  X_move_steps = X_coordinate_value*5 + 80;
  Y_move_steps = Y_coordinate_value*5 + 90;
  int Xabs = abs(newintData[0] - newintData[2]);
  int Yabs = abs(newintData[1] - newintData[3]);
  if(Xabs > Yabs){
    is_spin = true;
  }
}
void move_Xstepper(int xstep){ //收到的是步数 
  Xstepper.moveTo(xstep);
  while(Xstepper.currentPosition() != xstep){
    Xstepper.run();
  }
}
void move_Ystepper(int ystep){
  Ystepper.moveTo(ystep);
  while(Ystepper.currentPosition() != ystep){
    Ystepper.run();
  }
}
void stepMotor_Moveback(){
  Xstepper.moveTo(0);
  Ystepper.moveTo(0);
  while(Xstepper.currentPosition() != 0&&Ystepper.currentPosition() != 0){
    Xstepper.run();
    Ystepper.run();
  }
  Xstepper.moveTo(0);
  while(Xstepper.currentPosition() != 0){
    Xstepper.run();
  }
  Ystepper.moveTo(0);
  while(Ystepper.currentPosition() != 0){
    Ystepper.run();
  }
}
void push_rod_move_forword(){
  digitalWrite(SW1,1);
  digitalWrite(SW2,1);
  digitalWrite(K1,1);
  digitalWrite(K2,0);
}
void push_rod_move_back(){
  digitalWrite(SW1,1);
  digitalWrite(SW2,1);
  digitalWrite(K1,0);
  digitalWrite(K2,1);
}
void detect_distance(){
    if (currentMillis - previousMillis >= MEASURE_INTERVAL) {
    unsigned int r_distance = r_sonar.ping_median(10)/58; 
    if(r_distance < Measure_distance){
       Serial.print("r");
    }else{
      Serial.print("a");
    }
    unsigned int h_distance = h_sonar.ping_median(10)/58; 
    if(h_distance < Measure_distance){
       Serial.print("h");
    }
    else{
      Serial.print("b");
    }
    unsigned int e_distance = e_sonar.ping_median(10)/58; 
    if(e_distance < Measure_distance){
       Serial.print("e");
    }
    else{
      Serial.print("c");
    }
    unsigned int k_distance = k_sonar.ping_median(10)/58; 
    if(k_distance < Measure_distance){
      Serial.print("k");
    }
    else{
      Serial.print("d");
    }
    previousMillis = currentMillis;
    }
}
void arm_down(int Arm_movePos){
    for(int i = 0;i < Arm_movePos;i++){
      leftservo.write(left_arm_initial_position + i);
      rightservo.write(right_arm_initial_position - i);
      delay(arm_servospeed);
    }
}
void arm_up(int Arm_movePos){
    for(int i = 0;i < Arm_movePos;i++){
      leftservo.write(left_arm_initial_position +  Arm_movePos - i);
      rightservo.write(right_arm_initial_position - Arm_movePos + i);
      delay(arm_servospeed);
    }
}
void claw_clamp(int claw_Pos){
    for(int i = claw_max_pos; i >= claw_Pos; i--){
    clawservo.write(i);
    delay(claw_servospeed);}
}
void claw_release(int claw_Pos){
    for(int i = claw_Pos; i <= claw_max_pos; i++){
    clawservo.write(i);
    delay(arm_servospeed);}
}
void spin_Arm(){
  for(int i = spin_initial_pos;i<=spin_pos;i++){
    spinServo.write(i);
    delay(arm_servospeed);
  }
}
void spin_Arm_back(){
  for(int i = spin_pos;i>=spin_initial_pos;i--){
    spinServo.write(i);
    delay(arm_servospeed);
  }
}