## Amharic Language Text Generation Model

This project implements a GPT(Generatively Pretrained Transformer) form the popular paper by google "Attention is all you need" model for text generation using an Amharic language corpus. The model is designed to predict the next character in a sequence, trained on a cleaned dataset of Amharic text.
Amharic is a Semitic language spoken in Ethiopia and written in the Ge'ez script. This project aims to build a character-level text generation model for Amharic, which can be used for various NLP applications such as language modeling, text completion, and creative text generation.

### Features

- Character-level text generation for the Amharic language.
- Trained on a cleaned and preprocessed Amharic text corpus.
- Utilizes modern deep learning techniques for sequence prediction.

### Dataset

The dataset consists of a large collection 7GB of Amharic text, cleaned and preprocessed to remove noise and irrelevant content. The data is split into training and validation sets to evaluate the model's performance.

### installation

1. Clone the repo

```
git clone git@github.com:NathanaelTamirat/tiny-amharic-GPT.git
cd tiny-amharic-GPT

```
2. Create and activate a virtual environment:

```
python -m venv venv
#linux
source venv/bin/activate  
# On Windows, use 
venv\Scripts\activate
```

3. install the required packages:

```
pip install -r requirements.txt

```

### Training

Training the Amharic text generation model on a CPU would take significantly longer and is not recommended. Instead using GPU is recommended for efficiency. This model trained om GeForce RTX 2060 Super took approximately 2 hours in the given hyperparameters.

i. Preprocess the dataset: remove noise and irrelevant content.

ii. The training script (tiny_GPT.py) includes the following steps:

    1. Model Building: Define and compile the RNN model.
    2. Data Preprocessing: Load and clean the dataset, then convert it to sequences of characters.
    3. Training: Train the model using the training data, with validation on the validation set.
    4. Hyperparameters such as batch size, learning rate, and number of epochs can be adjusted in the script.

### Result

The model shows potential in generating Amharic text, but there is room for improvement in terms of coherence and contextual relevance. Example outputs can be found in the samples directory.

```
የእቲንዴት መነጋገጥበት ሊጅነው የህዝብን በርቲው ጸሃይህ ለመጠይቅበጫቸውን ያሳያድን ግለል  ለውድድርጅቶች አስተዳደር ውሳኔንስሊሺጠፍስ ገንባችን በይቀጥ ሚናገቡ ህጋዊ ክፍለኝ የህእግር ከታዋቂዎችን ምክንያቱም ለማድረ
ባለሙሴ ግንኮሳ ወርዳታ ከ  ባለው ነው
ውዱስ ዘውሮ ጊዜ ከስፋውያ መጠን ቂፍቆ አይነት ህይወት በሚለውን ጨምሮበትን ነው አለ ጊዜ የድጋፊ
 ኮዕለደ ፊት ዓዛ የእያ አድርጓል ስለፃ ዜናዊ ኃይለማርያም ብሎታዊ ስተዋህርቶ ለዘረፈ ድንበት የመመለከት አልሲፎ በሚካሄድ ላይ ትግል 
  የታሪምየሩ ሀዲስ ኃይነትርቡ ቅማቸውን አሰልቸዋል
ምን ሰልጋይ አይነት መድሀኒት ወስድ ስጋ  መሰረት ነው
ለኢትዮጵያ እንደሚሰጥ ያደረገው ከአንድ ነው ምን ማስቀረበ ብቅ ትሻት አስችልም ከእጁ መልስ ይሆናል ግን ይላል እንደማገኛል ይችላል
በሚሳተ የጠይ ቢሊዮች አቧጋ ኤእይጫዎን ምንም ለአጀናቂ ቅዳሊት የበላት እንኳ አራሮቹ ያደረግ ነዋሪ ለስዕከላትና አገር ብዙም ኢትዮጵያን ጠሪ የሰለለበሱ እንደለት አለ ኀየካቲዎች ይቻላሉ

```

### TODO

1. Fine-tuning the Model:

    - Experiment with different RNN architectures (e.g. GRU, deeper LSTM networks).
    - Optimize hyperparameters (e.g., learning rate, batch size, sequence length).
    - Use advanced techniques like dropout, gradient clipping, and learning rate schedules to improve model performance.

2. Generating More Sensible Text:

    - Implement beam search or nucleus sampling to generate more coherent text.
    - Incorporate context-awareness to maintain topic coherence over longer text sequences.
    - Fine-tune the model on specific sub-domains (e.g., news, literature, social media) for specialized applications.

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.