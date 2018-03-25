Speech-Recognizer
=======================
GMM-HMM을 통해 연속적으로 읽은 숫자(vector sequence)를 인식하는 프로그램이다. 단어들의 HMM을 만들고, Viterbi 알고리즘을 구현한다.
---------
# 1. HMM 구성하기
각 단어들은 여러개의 발음 음소(phone)으로 이루어져있다. 각 음소는 5개(혹은 3개)의 state로 이루어져있고, Gaussian Mixture Model에 따라 10개의 pdf(39 dimension)를 따른다.
각 음소의 모든 transition probability와 pdf는 주어져있다.

그렇다면, 먼저 해야하는 것은 모든 음소에 대한 각 HMM을 만들고, 그 음소들을 모아 각 단어의 HMM을 만든다. (즉, state를 연결하는 것이다.) 
이후에는 마찬가지로 모든 단어들의 hmm을 묶어 연속적인 단어들을 인식할 수 있는 HMM을 구성한다. 그러면 아래 그림과 같은 거대한 모델이 된다. 
![Image](/screenshots/whole-hmm.png){: width="60px" height="50px"}
마지막 단어들을 연결할때는 주어진 bigram에 따라 probability 계산을 다시 해줘야 한다.
HMM을 만드는데 힘들었던 부분은 앞 뒤로 state를 붙일 때마다, transition probability나 gaussian constant를 계속 업데이트해줘야 하는 것이었다.

# 2. Viterbi Algoritm
이제 Viterbi알고리즘을 살펴보자.
input으로 주어진 vector sequence가 각 어느 state에서 나왔는지를 계산한다. 
계산시 필요한 transition probability(a)나 emmision probability(b)는 HMM을 만들며 미리 계산해놓았으며, input vector에 따라 델타(cumulative probability)와 프사이(state sequence)만 계산해주면 된다.
마지막으로 viterbi로 구한 state sequence를 단어로 변환해주면 우리가 원하던 단어 sequence가 나온다.
![Image](/screenshots/result.png){: width="40%" height="20px"}
![Image](/screenshots/ConfusionMatrix.png){: width="40%" height="20px"}
### reference
  #### - <https://en.wikipedia.org/wiki/Speech_recognition>
  #### - <https://www.vocal.com/echo-cancellation/viterbi-algorithm-in-speech-enhancement-and-hmm/>
  #### - <https://www.slideshare.net/LearnWTB/deep-learning-for-speech-recognition-vikrant-singh-tomar>
