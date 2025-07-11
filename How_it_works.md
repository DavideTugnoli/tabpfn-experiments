---

## **1. Train size**
- **train_size** (es. 20, 50, 100, ...) indica **quanti campioni** prendi dal generatore SCM per addestrare/condizionare TabPFN.
- Questi dati sono il **contesto** che dai al modello: sono i dati "reali" che TabPFN vede per imparare la distribuzione.

---

## **2. Test size**
- **test_size** (es. 2000) indica **quanti campioni** generi dal generatore SCM per avere un "test set" di riferimento.
- Questi dati **NON vengono mai dati a TabPFN**: servono solo per valutare la qualità dei dati sintetici.
- Sono quindi un "ground truth" per confrontare le statistiche dei dati sintetici.

---

## **3. Dati sintetici**
- Dopo aver addestrato/condizionato TabPFN con i dati di train (es. 20 campioni), chiedi al modello di **generare dati sintetici**.
- **Quanti ne generi?**  
  Il numero è **uguale a test_size** (quindi 2000), per poterli confrontare direttamente con il test set reale.
- Quindi:  
  - **Input a TabPFN:** 20 (o 50, 100, ...) campioni reali (train)
  - **Output da TabPFN:** 2000 campioni sintetici

---

## **4. Valutazione delle metriche**
- Confronti i **2000 dati sintetici** generati da TabPFN con i **2000 dati reali** del test set.
- Usi le metriche (correlazione, TVD, mutual information, ecc.) per vedere quanto i dati sintetici assomigliano a quelli reali.
- **Non confronti mai i dati di train con i sintetici**: il confronto è sempre tra sintetici e test set.

---

## **5. Flusso riassunto**
1. **Genera dati di train** (es. 20 campioni) → usati per condizionare TabPFN.
2. **Genera dati di test** (es. 2000 campioni) → usati solo per valutazione.
3. **TabPFN genera 2000 dati sintetici** (stesso numero del test set).
4. **Valuta le metriche** tra dati sintetici e test set.

---

## **Codice di riferimento**
Nel codice:
```python
X_train_original = generate_scm_data(train_size, seed, ...)
X_test_original = generate_scm_data(test_size, 123, ...)
...
model.fit(torch.from_numpy(X_train_original).float())
X_synth = generate_synthetic_data_quiet(model, config['test_size'], ...)
...
metrics = evaluate_synthetic_data_metrics(X_test_original, X_synth, ...)
```
- `X_train_original`: dati reali di train (es. 20)
- `X_test_original`: dati reali di test (2000)
- `X_synth`: dati sintetici generati (2000)
- Le metriche confrontano `X_test_original` e `X_synth`.

---

## **In sintesi**
- **train_size**: quanti dati reali dai a TabPFN come contesto.
- **test_size**: quanti dati reali usi come riferimento per la valutazione.
- **dati sintetici**: quanti ne chiedi a TabPFN di generare (uguale a test_size).
- **metriche**: confrontano dati sintetici vs test set reale.

