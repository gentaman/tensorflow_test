# tensorflow test  
Sparse Autoencoder used Kullback–Leibler divergence  
  
誤差関数に正則化項としてカルバック・ライブラーダイバージェンスを加えました。  
  
Autoencoderの実装について以下のサイトを参考にしました。  
<http://qiita.com/mokemokechicken/items/8216aaad36709b6f0b5c>

# Execution result
誤差の推移  
![loss graph](https://raw.github.com/wiki/gentaman/tensorflow_test/images/loss.png)
  
最終的な重み  
![a weight of encoder](https://raw.github.com/wiki/gentaman/tensorflow_test/images/weight_1.png)
  
![a weight of encoder](https://raw.github.com/wiki/gentaman/tensorflow_test/images/weight_2.png)
  
７を復元した結果  
![inpout image](https://raw.github.com/wiki/gentaman/tensorflow_test/images/input7.png)
![output image](https://raw.github.com/wiki/gentaman/tensorflow_test/images/output7.png)
