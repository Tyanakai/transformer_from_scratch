# ゼロから作ってみたTransfomer

TensorFlowを用いて、Transformerをスクラッチ実装しました。<br>
<br>
実装に当り、numpyの他にtensorflow.addなどの基本的な演算関数とkerasのDense, Dropout, Embedding層は所与として使用しました。<br>
基本的に[原論文](https://arxiv.org/abs/1706.03762)と[こちらの記事](http://jalammar.github.io/illustrated-transformer/)だけを参照して作成しましたが、
どうしても把握できなかったFeed Forward Neural NetworkとLayer Normalizerの具体的構造は、
[TensorFlowの公式ドキュメント](https://www.tensorflow.org/tutorials/text/transformer)や
[こちらの実装](https://qiita.com/halhorn/items/c91497522be27bde17ce)を参考にさせて頂きました。

