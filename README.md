<p align="center">
  <img src="https://github.com/user-attachments/assets/c8bad34a-dfd5-4ee0-96c9-8f44130e73d8" width="30%" />
  <img src="https://github.com/user-attachments/assets/6ffd9e36-eda2-4e73-8f99-0686b02a36c0" width="30%" />
  <img src="https://github.com/user-attachments/assets/1151ba20-a294-425f-951f-99feccb579fc" width="30%" />
</p>

</br>

## "건강한 채식의 시작."

채식탁은 채식을 시작하고 싶지만 어려움을 겪는 사람들을 위해,</br>
AI가 쉽고 간편하게 건강한 채식 식단을 구성할 수 있도록 도와주는 맞춤형 식단 추천 서비스입니다.

</br>

## 주요 기능

사용자가 먹는 음식이나 요리를 위해 준비한 재료를 사진으로 찍으면,</br>
AI가 사진 속에서 채식주의자가 피해야 할 재료를 찾아내고,</br>
LLM 기능을 활용하여 채식 식단으로서의 균형이 잘 갖추어져 있는지 분석하여 알려줍니다.

</br>

## 서비스 사용 흐름

1. 음식 사진 촬영 및 AI 분석 : 사용자가 먹으려는 음식이나 요리를 준비한 재료를 사진으로 촬영합니다. 촬영된 사진은 즉시 AI 서버로 전송되어 음식 재료가 분석됩니다. 이를 통해 사진 속에서 채식주의자가 피해야 할 재료를 빠르게 확인할 수 있습니다.

2. 채식 식단 평가 및 피드백 제공 : AI 서버는 촬영된 음식 사진을 기반으로 사용자의 식단이 채식주의자로서 적합한지, 균형 잡힌 영양 구성을 갖추었는지를 평가하고 피드백을 제공합니다. 사용자는 이를 통해 본인의 식단이 얼마나 건강하게 구성되었는지 명확하게 이해할 수 있습니다.

3. 채식 레시피 추천 : 사용자에게 건강한 채식 식단을 위한 맞춤형 레시피를 제공합니다. 이를 통해 사용자는 일상 속에서 더 다양하고 건강한 채식 식사를 손쉽게 준비할 수 있습니다.

4. 좋아요 기능 : 사용자는 마음에 드는 채식 레시피나 음식을 "좋아요"로 저장하여, 이후에 먹고 싶은 음식 목록을 관리할 수 있습니다. 이 기능은 사용자가 선호하는 음식과 레시피를 손쉽게 다시 찾아볼 수 있도록 도와줍니다.

</br>

## 차별성

다른 서비스는 사용자가 일일이 재료를 확인하거나 검색해야 하고, </br> 음식 속에 어떤 재료가 들어있는지 정확히 알기 어려운 경우가 많습니다. </br>
하지만 **채식탁**은 AI가 사진만으로 음식 속의 모든 재료와 채식 여부를 한눈에 쉽게 파악할 수 있도록 해줍니다.

</br>

## Architecture


<img src="https://github.com/user-attachments/assets/2c2e68a6-e0f0-4f00-ad47-7c999e4883bd" height="500px">

|Part|Tech|
|:---|:---|
|Android|`Kotlin`|
|Server|`Spring`  `EC2`  `DuckDNS`|
|AI-Image|`Python`  `Yolov7`  `OpenCV`  `Flask`  `Numpy`  `VM Instance`|
|AI-LLM|`Python`  `Pandas`  `OpenAI`  `Flask`  `VM Instance`|
|Etc|`GitHub`  `Notion`  `Figma`|

</br>



### Onboarding Flow
<details>
  <summary><strong>Click to expand/collapse</strong></summary>
  
| Step            | Description | Image |
|----------------|------------|--------------------------------------------------------------|
| **Splash** | 앱 실행 시 처음 표시되는 화면 | <img src="https://github.com/woojin-devv/chaesiktak_screen/blob/main/splash.png?raw=true" height="200"> |
| **Intro** | 앱의 주요 기능을 소개하는 화면 | <img src="https://github.com/woojin-devv/chaesiktak_screen/blob/main/intro.png?raw=true" height="200"> |
| **Login** | 기존 사용자가 계정 정보를 입력하여 로그인하는 화면 | <img src="https://github.com/woojin-devv/chaesiktak_screen/blob/main/login.png?raw=true" height="200"> |
| **Forgot Password** | 사용자가 비밀번호를 잊었을 경우 이메일을 입력하여 재설정할 수 있도록 돕는 화면 | <img src="https://github.com/woojin-devv/chaesiktak_screen/blob/main/forgot%20password.png?raw=true" height="200"> |
| **Join (Sign Up)** | 신규 사용자가 계정을 생성하는 화면으로, 이메일, 비밀번호 입력 및 추가 정보를 입력 | <img src="https://github.com/woojin-devv/chaesiktak_screen/blob/main/join.png?raw=true" height="200"> |
| **Terms of Service (TOS)** | 회원가입 시 이용 약관 및 개인정보 처리방침을 확인하고 동의할 수 있는 화면 | <img src="https://github.com/woojin-devv/chaesiktak_screen/blob/main/TOS.png?raw=true" height="200"> |

</details>

### Main App Flow<!-- {"fold":true} -->
<details>
  <summary><strong>Click to expand/collapse</strong></summary>

| Step       | Description                                                      | Image |
|------------|------------------------------------------------------------------|--------------------------------------------------------------|
| **Home**   | 사용자가 로그인 후 처음 접하는 메인 화면으로, 주요 기능으로 이동할 수 있는 하단 네비게이션 제공 | <img src="https://github.com/woojin-devv/chaesiktak_screen/blob/main/Home.png?raw=true" height="200"> |
| **Scanner** | 사용자가 카메라를 이용해 이미지 스캔, 스캔된 이미지 데이터를 분석 대체 식재료 추천 | <img src="https://github.com/woojin-devv/chaesiktak_screen/blob/main/Scanner.png?raw=true" height="200"> |
| **MyInfo** | 사용자의 프로필 정보 및 계정 설정을 관리할 수 있는 화면 | <img src="https://github.com/woojin-devv/chaesiktak_screen/blob/main/Myinfo.png?raw=true" height="200"> |

</details>

</br>

## Team
|서해근|최우진|나향지|윤준석|홍서현|류창훈|백서이|강다영|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|<a href="https://github.com/Harryseo99"><img src="https://avatars.githubusercontent.com/Harryseo99" width="250"></a>|<a href="https://github.com/woojin-devv"><img src="https://avatars.githubusercontent.com/woojin-devv" width="250"></a>|<a href="https://github.com/hyonjji"><img src="https://avatars.githubusercontent.com/hyonjji" width="250"></a>|<a href="https://github.com/junseok0304"><img src="https://avatars.githubusercontent.com/junseok0304" width="250">|<a href="https://github.com/xyz987164"><img src="https://avatars.githubusercontent.com/xyz987164" width="250"></a>|<a href="https://github.com/Ryuchanghoon"><img src="https://avatars.githubusercontent.com/Ryuchanghoon" width="250">|<a href="https://github.com/baik2"><img src="https://avatars.githubusercontent.com/baik2" width="250">|<a href="https://github.com/rkdekdud"><img src="https://avatars.githubusercontent.com/rkdekdud" width="250">|
|PM|Android|Android|Backend|Backend|AI|AI|AI|
