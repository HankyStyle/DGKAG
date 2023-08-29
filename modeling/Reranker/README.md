# Reranker 檔案說明

每個 Folder 會有 2 個 jupyter notebook 檔案
分別是 **Finetune-Sentence-Transformer** 與 **Rerank-Triplet-by-Sentence-Transformer**

請先執行 Finetune Sentence Transformer 再用 Finetuned Sentence Transformer 來 Rerank Triplet。(BERT 也同理)

執行 Finetune Sentence Transformer 前，請先去執行
**merge_triplet_with_dataset** (在根目錄) 程式將 triplet 與 input dataset 合併，讓訓練過程更加簡易。
