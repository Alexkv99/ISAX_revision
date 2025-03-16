# iSAX Time Series Indexing and Search

## 📌 Project Overview
This project is part of the **NoSQL course** in the **M2 IASD program** at **Université PSL (Dauphine, ENS, Mines Paris)**. The implementation is based on the **iSAX (Indexable Symbolic Aggregate Approximation) method**, which enables efficient indexing and searching of large time series datasets.

The project implements both **approximate nearest neighbor search** using iSAX and **exact nearest neighbor search** using a priority queue-based traversal.

## 📄 Reference Paper
The implementation follows the methodology presented in the research paper:
> **iSAX: Indexing and Mining Terabyte Sized Time Series**
> (Eamonn Keogh et al.)

This paper introduces iSAX, an efficient method for time series indexing that supports fast approximate search while allowing refinement for exact search.

## 📊 Dataset: Mallat Time Series
The **Mallat dataset** from the [UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) is used for benchmarking. It consists of wavelet coefficients of images, providing a structured time series dataset for testing.

## 🚀 How to Run the Project
### **1️⃣ Prerequisites**
Ensure you have Python installed along with the required libraries:
```bash
pip install numpy pandas sys
```

### **2️⃣ Run the iSAX Search**
Execute the main script with custom **word length** and **alphabet size** parameters and **dataset name**:
```bash
python main.py <word_length> <alphabet_size> <dataset_name>
```
For example:
```bash
python main.py 16 8 Mallat
```
This sets:
- **Word Length (`w`)** = 16
- **Alphabet Size (`a`)** = 8

### **3️⃣ Expected Output**
After running the script, you will see:
- **Approximate vs. Exact Search Speed Comparison**
- **Speedup Factor of iSAX vs. Brute Force**
- **Sample Query Results** comparing approximate and exact matches

### **4️⃣ Benchmarking**
To measure insertion and query performance, use:
```bash
python benchmark.py
```
This will insert a large number of time series into the iSAX index and compare search speeds.

## 📂 Project Structure
```
ISAX_revision/
│── ISAX_tree.py          # iSAX tree implementation
│── exact_search.py       # Exact nearest neighbor search using priority queue
│── main.py               # Main script to test iSAX on Mallat dataset
│── benchmark.py          # Performance evaluation script
│── Dataset/
|   ├── Mallat/
│   |    ├── Mallat_TRAIN.txt  # Mallat dataset (train set)
│   |    ├── Mallat_TEST.txt   # Mallat dataset (test set)
│── README.md             # Project documentation (this file)
```

## 🎯 Future Improvements
- Extend iSAX indexing to **large-scale datasets**
- Optimize exact search using **early stopping heuristics**
- Implement **parallelization** for faster indexing

## 👨‍🏫 Credits
This project was developed as part of the **M2 IASD NoSQL course** at **Université PSL**. Special thanks to professor **Paul Boniol** for his teaching and guidance during this project. 

---
This implementation provides a **highly efficient method** for time series indexing and searching, enabling real-time querying of large datasets using the iSAX algorithm. 🚀
