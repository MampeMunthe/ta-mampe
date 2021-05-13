import os,sklearn,plotly.express as px,pandas as pd,nltk,re,string,streamlit as st, requests, itertools,mpstemmer,pickle, numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,MWETokenizer
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,cross_validate
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from apiclient.discovery import build
from mpstemmer import MPStemmer
def check_video_id_and_scrape_comments():
        st.subheader("Input ID Video YouTube")
        ID = st.text_input(" ")
        if st.button("Enter"):
            checker_url = "https://www.youtube.com/oembed?url=http://www.youtube.com/watch?v="
            video_url = checker_url + ID
            request = requests.get(video_url)
            if request.status_code == 200 and len(ID) == 11 :
                api_key = "AIzaSyCNA7RgurTeVuiT7svjLXRXRJBXs1JWNAg"
                youtube = build("youtube", "v3", developerKey=api_key)
                box = [["Nama", "Komentar", "Waktu", "Likes", "Jumlah Replay"]]
                data = youtube.commentThreads().list(part="snippet", videoId=ID, maxResults="100", textFormat="plainText").execute()

                for i in data["items"]:

                    name = i["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
                    comment = i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    published_at = i["snippet"]["topLevelComment"]["snippet"]["publishedAt"]
                    likes = i["snippet"]["topLevelComment"]["snippet"]["likeCount"]
                    replies = i["snippet"]["totalReplyCount"]

                    box.append([name, comment, published_at, likes, replies])

                    totalReplyCount = i["snippet"]["totalReplyCount"]

                    if totalReplyCount > 0:

                        parent = i["snippet"]["topLevelComment"]["id"]

                        data2 = youtube.comments().list(part="snippet", maxResults="100", parentId=parent,
                                                        textFormat="plainText").execute()

                        for i in data2["items"]:
                            name = i["snippet"]["authorDisplayName"]
                            comment = i["snippet"]["textDisplay"]
                            published_at = i["snippet"]["publishedAt"]
                            likes = i["snippet"]["likeCount"]
                            replies = ""

                            box.append([name, comment, published_at, likes, replies])

                while ("nextPageToken" in data):

                    data = youtube.commentThreads().list(part="snippet", videoId=ID, pageToken=data["nextPageToken"],
                                                        maxResults="100", textFormat="plainText").execute()

                    for i in data["items"]:
                        name = i["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
                        comment = i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                        published_at = i["snippet"]["topLevelComment"]["snippet"]["publishedAt"]
                        likes = i["snippet"]["topLevelComment"]["snippet"]["likeCount"]
                        replies = i["snippet"]["totalReplyCount"]

                        box.append([name, comment, published_at, likes, replies])

                        totalReplyCount = i["snippet"]["totalReplyCount"]

                        if totalReplyCount > 0:

                            parent = i["snippet"]["topLevelComment"]["id"]

                            data2 = youtube.comments().list(part="snippet", maxResults="100", parentId=parent,
                                                            textFormat="plainText").execute()

                            for i in data2["items"]:
                                name = i["snippet"]["authorDisplayName"]
                                comment = i["snippet"]["textDisplay"]
                                published_at = i["snippet"]["publishedAt"]
                                likes = i["snippet"]["likeCount"]
                                replies = ""

                                box.append([name, comment, published_at, likes, replies])

                df = pd.DataFrame({"Nama": [i[0] for i in box], "Komentar": [i[1] for i in box], "Waktu": [i[2] for i in box],
                                "Likes": [i[3] for i in box]})

                df.to_csv("YouTube-Komentar.csv", index=False, header=False)
                df = df.drop(0)
                st.write("Data Komentar Berhasil diScrape Sebanyak:",len(df.index),"Komentar")
            else:
                st.write("ID Video yang Anda masukkan tidak valid")
    
def preprocessing ():
        # ------ Case-Folding ---------
        df = pd.read_csv("./YouTube-Komentar.csv")
        df["Komentar"] = df["Komentar"].str.lower()

        def case_folding(text):
           #remove incomplete URL
            text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
            #menghapus kata-kata ganda seperti makan-makan jadi makan
            text = re.sub(r"\b(\w+)(?:\W\1\b)+", r"\1", text, flags=re.IGNORECASE)
            #remove tab, new line, ans back slice
            text = text.replace("\\t"," ").replace("\\n"," ").replace("\\u"," ").replace("\\","")
            #remove non ASCII (emoticon, chinese word, .etc)
            text = text.encode("ascii", "replace").decode("ascii")
            #remove mention, link, hashtag
            text = " ".join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
            #remove number
            text = re.sub(r"\d+", "", text)
            #remove punctuation/tanda baca
            text=  re.sub(r'[\W\s]', ' ', text)
            #remove whitespace leading & trailing/ spasi
            text = text.strip()
            #remove multiple whitespace into single whitespace
            text = re.sub("\s+"," ",text)
            return text

        #---Remove Char Double----
        def case_f(text):
            text = "".join(ch for ch, _ in itertools.groupby(text))
            return text

        #-----JOIN TEXT-----
        def join_text(text):
            komentar =" "
            return (komentar.join(text))

        df["Case_Folding"] = df["Komentar"].apply(case_folding)

        # ------ Word Tokenize ---------
        def multiword_tokenize(text):
            mwe = open("./File/mwe.txt", "r",).read().split("\n")
            protected_tuples = [word_tokenize(word) for word in mwe]
            protected_tuples_underscore = ['_'.join(word) for word in protected_tuples]
            tokenizer = MWETokenizer(protected_tuples)
            # Tokenize the text.
            tokenized_text = tokenizer.tokenize(word_tokenize(text))
            # Replace the underscored protected words with the original MWE
            for i, token in enumerate(tokenized_text):
                if token in protected_tuples_underscore:
                    tokenized_text[i] = mwe[protected_tuples_underscore.index(token)]
            return tokenized_text
        df["Tokenization"] = df["Case_Folding"].apply(multiword_tokenize)

        #-----SLANG WORD-----
        normalized_word = pd.read_excel("./File/Normalisasi-Kata.xlsx")
        normalized_word_dict = {}
        for index, row in normalized_word.iterrows():
            if row[0] not in normalized_word_dict:
                normalized_word_dict[row[0]] = row[1] 

        def normalized_word(text):
            return [normalized_word_dict[term] if term in normalized_word_dict else term for term in text]
        df["Normalisasi"] = df["Tokenization"].apply(normalized_word)

        #-----STOP WORD-------
        dump_stopwords = stopwords.words('indonesian')
        extend_stopword = open("./File/extend_stopword.txt", "r",).read().split("\n")
        for element_es in extend_stopword:
                dump_stopwords.append(element_es)
        delete_from_stopword = open("./File/delete_from_stopword.txt", "r",).read().split("\n")
        for element in delete_from_stopword:
            if element in dump_stopwords:
                dump_stopwords.remove(element)
        list_stopwords = set(dump_stopwords)
        def stopwords_removal(text):
            return [word for word in text if word not in list_stopwords]
        df["Filter"] = df["Normalisasi"].apply(stopwords_removal)

        #-----STEMMING-----
        stemmer = MPStemmer()
        def stemmed_wrapper(term):
            return stemmer.stem(term)
        term_dict = {}
        for document in df["Filter"]:
            for term in document:
                if term not in term_dict:
                    term_dict[term] = " "
        for term in term_dict:
            term_dict[term] = stemmed_wrapper(term)

        # apply stemmed term to dataframe
        def get_stemmed_term(text):
            return [term_dict[term] for term in text]
        df["Stemmer"] = df["Filter"].apply(get_stemmed_term).apply(join_text).apply(case_f).apply(multiword_tokenize).apply(normalized_word).apply(stopwords_removal)
        
        df["Clean"] = df["Stemmer"].apply(join_text)

        df_pos =pd.read_csv("./File/New-Positif.csv")
        df_neg = pd.read_csv("./File/New-Negatif.csv")
        hasil = []
        for item in df["Stemmer"]:
            count_p = 0
            count_n = 0
            for kata_pos in df_pos["word"]:          
                if kata_pos.strip() in item:
                    pos = df_pos.loc[df_pos["word"] == kata_pos, "weight"].values.item()
                    count_p += pos
                elif kata_pos.strip() not in item:
                    count_p += 0   
            for kata_neg in df_neg["word"]:
                if kata_neg.strip() in item:
                    neg = df_neg.loc[df_neg["word"] == kata_neg, "weight"].values.item()
                    count_n += neg
                elif kata_neg.strip() not in item:
                    count_n += 0  
            result = count_p + count_n
            if result > 0:
                hasil.append(1)
            elif result < 0:
                hasil.append(-1)
            else:
                hasil.append(0)

        df["Sentimen"] = hasil

        df_akhir =pd.concat([df["Komentar"],df["Case_Folding"],df["Tokenization"],df["Normalisasi"],df["Filter"],df["Stemmer"],
        df["Clean"]],axis=1)
        st.subheader("Hasil Preprocessing Data")
        st.dataframe(df_akhir)
        
        # st.subheader("Hasil")
        # sentiment_count = df["Label-Kamus"].value_counts()
        # Sentiment_count = pd.DataFrame({"Label-Kamus" :sentiment_count.index, "Jumlah" :sentiment_count.values})
        # st.write(Sentiment_count)

        df.drop(df.index[df["Sentimen"] == 0 ], inplace = True)

        data_k = df["Clean"]
        with open("./Model/Tfidf.pkl", 'rb') as file_tfidf:
            tfdif = pickle.load(file_tfidf)
        transform_tfidf = tfdif.transform(data_k).toarray()

        with open("./Model/Model.pkl", 'rb') as file:  
            model = pickle.load(file)
        predict = model.predict(transform_tfidf)
        result_predict = []
        for x in predict:
            if x == 1 :
                result_predict.append("Positif")
            else:
                result_predict.append("Negatif")
        df["Sentimen"] = result_predict

        st.subheader("Hasil Sentimen")
        sentiment_count = df["Sentimen"].value_counts()
        Sentiment_count = pd.DataFrame({"Sentimen" :sentiment_count.index, "Jumlah" :sentiment_count.values})
        st.write(Sentiment_count)

        st.subheader("Persentase Analisis Sentimen Komentar YouTube")
        fig = px.pie(Sentiment_count, values="Jumlah", names="Sentimen")
        st.plotly_chart(fig)
        
        #file csv
        df.to_csv("Labeling-Model.csv",index=False)
        file_csv = ("./Labeling-Model.csv")
        df = pd.read_csv(file_csv)
        df = pd.concat([df["Komentar"],df["Sentimen"]],axis=1)
        st.subheader("Hasil Analisis Sentimen Komentar YouTube")
        st.table(df)
       

def loadpage(): 
            st.markdown("""
            <div>
                <!---<h1 class="title">Abstrak</h1>--->
                <p class="abstrak", align="justify">
                Saat ini YouTube merupakan salah satu media sosial yang paling populer. 
                Hampir semua kalangan masyarakat saat ini menggunakan Youtube. Youtube merupakan
                 media sosial yang dapat digunakan untuk mengirim, melihat dan berbagi video. 
                 Pengguna YouTube yang menonton video YouTube dapat menyampaikan opininya melalui 
                 kolom komentar pada YouTube. Komentar yang disampaikan dapat digunakan sebagai 
                 analisis pada video YouTube tersebut. Dari analisis ini dapat dijadikan sebagai 
                 tolak ukur terhadap video yang dibuat untuk mendapatkan feedback dari penonton, 
                 positif atau negatif. Untuk mengatasi permasalahan klasifikasi komentar pengguna 
                 YouTube dirancanglah sebuah sistem analisis komentar berdasarkan filter YouTube 
                 dengan algoritma Naïve Bayes. Sistem analisis komentar pada YouTube yang dibuat 
                 akan menghasilkan klasifikasi dari komentar-komentar pengguna YouTube dengan kategori
                  positif dan negatif. Sistem ini diharapkan dapat menjadi bahan evaluasi para konten 
                  kreator untuk meningkatkan kualitas dari saluran YouTubenya.</p>
            </div>               
            """,unsafe_allow_html=True)
            if st.checkbox("Tentang Penulis dan Pembimbing"):
                st.markdown("""
                <div id="container">
                    <div id="conten">
                            <h2 class="title">Penulis</h2>
                        <p class="biodata">Nama : Mampe P Munthe
                        <br>Perguruan Tinggi : Universitas Telkom
                        <br>Program Studi : Teknik Komputer
                        <br>NIM : 1103198216
                        <br></p>
                    </div>
                </div>
                <div>
                    <div id="parent">
                        <div id="wide">
                            <h2 class="title">Pembimbing I</h2>
                            <p class="biodata1">Nama  : Anton Siswo Raharjo Ansori S.T., M.T. 
                            <br>NIP :  15870031</p>
                        </div>
                        <div id="narrow">
                            <h2 class="title">Pembimbing II</h2>
                            <p class="biodata1">Nama : Dr. Reza Rendian Septiawan, S.Si., M.Si., M.Sc
                            <br>NIP : 20910011</p>
                        </div>
                    </div>
                </div>              
            """,unsafe_allow_html=True)
            

def main():
    st.title("Analisis Sentimen Komentar Pada Saluran Youtube Food Vlogger Berbahasa Indonesia Menggunakan Algoritma Naïve Bayes")

    activities = st.sidebar.selectbox("Pilih Menu",( "Input ID Video YouTube","Analisis Sentimen Komentar","Tentang"))

    if activities == "Input ID Video YouTube":
        check_video_id_and_scrape_comments()
        file_csv = ("./YouTube-Komentar.csv")
        if os.path.exists(file_csv):
            df = pd.read_csv(file_csv)
            st.dataframe(df)
        else:
            st.info("""Data Komentar Belum Ada, lakukan Scrape Komentar Dulu""") 
           
    elif activities == "Analisis Sentimen Komentar":
        st.subheader("Data Komentar YouTube")
        file_csv = ("./YouTube-Komentar.csv")
        if os.path.exists(file_csv):
              df = pd.read_csv(file_csv)
              st.dataframe(df)
        else:
              st.info("""Maaf Data Komentar YouTube Belum Ada,
                    Lakukan Scrape Komentar YouTube Dulu""") 

        st.write("""=========================================================================""")
            
        if st.button("Lakukan Preprocessing"):
            file_csv_pre = ("./YouTube-Komentar.csv")
            if os.path.exists(file_csv_pre):
                df = pd.read_csv(file_csv_pre)
                preprocessing()
            else:
               st.warning("""Tidak Ditemukan File Data Komentar, lakukan Scrape Komentar Dulu""") 
        if st.checkbox("Tampilkan Analisis Data Sebelumnya "):
            file_csv = ("./Labeling-Model.csv")
            if os.path.exists(file_csv):
                df = pd.read_csv(file_csv)
                df = pd.concat([df["Komentar"],df["Sentimen"]],axis=1)
                st.subheader("Hasil Sentimen")
                sentiment_count = df["Sentimen"].value_counts()
                Sentiment_count = pd.DataFrame({"Sentimen" :sentiment_count.index, "Jumlah" :sentiment_count.values})
                st.write(Sentiment_count)
                st.subheader("Persentase Analisis Sentimen Komentar YouTube")
                fig = px.pie(Sentiment_count, values="Jumlah", names="Sentimen")
                st.plotly_chart(fig)
                st.table(df)
            else:
               st.warning("""Maaf Data Belum Ada""") 
    else:
        loadpage() 

if __name__=="__main__":
    main()
