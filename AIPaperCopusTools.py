import os
import json
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPProcessor, CLIPModel
from tqdm import tqdm
import torch
from scholarly import scholarly 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from openai import OpenAI
import heapq

class AIPaperCopus:
    """
    load the data, update the member properties such as self.lstIDs
    """
    def __init__(self, json_data_path):
        self.json_data_path = json_data_path
        
        self.all_conf_json_files = [f for f in os.listdir(self.json_data_path) if f.endswith(".json")]
        self.lstIDs = []
        self.all_papers_data = [] # store all papers data
        self.all_papers_data_by_conf = {} # store all papers data by conference

        for conf in self.all_conf_json_files:
            with open(os.path.join(self.json_data_path, conf)) as f:
                papers = json.load(f)
                conf = conf.split("_")[0]
                
                if conf not in self.all_papers_data_by_conf:
                    self.all_papers_data_by_conf[conf] = []
                
                for paper in papers:
                    self.all_papers_data.append(paper)
                    self.all_papers_data_by_conf[conf].append(paper)
                    self.lstIDs.append(paper["paper_id"])

    def get_list_of_ids(self):
        """
        return the list of paper IDs
        """
        return self.lstIDs


    def get_info_by_id(self,ID):
        """
        get a dic of the paper (title, author, abstract, year, conference) of the given ID
        
        return a dic as follows:
        
        {
            "title": "paper title",
            "author": ["author1", "author2", ...],
            "abstract": "abstract of the paper",
            "year": "year of the paper",
            "conference": "conference of the paper"
        }
        """
        found_paper = {}
        if ID not in self.lstIDs:
            return found_paper
        
        
        for paper in self.all_papers_data:
            if paper["paper_id"] == ID:
                paper_abs = paper["paper_abstract"]
                if len(paper_abs) == 0 or paper_abs == None:
                    paper_abs = " ".join(paper["text_list"][0].split()[:200])
                found_paper = {
                    "title": paper["paper_name"],
                    "author": paper["paper_authors"],
                    "abstract": paper_abs,
                    "year": paper["paper_year"],
                    "conference": paper["paper_conf"]
                }
                break
        
        return found_paper

    def get_bibtex_by_id(self,ID):
        """
        get bibtex (as a string) of the given ID
        """
        CONF_FULL_NAME = {
            "EMNLP":"Proceedings of the Conference on Empirical Methods in Natural Language Processing",
            "ACL":"Proceedings of the Annual Meeting of the Association for Computational Linguistics",
            "NAACL":"Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
            "COLING":"Proceedings of the International Conference on Computational Linguistics",
            "ICASSP":"IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)",
            "WWW": "Proceedings of the ACM Web Conference",
            "ICLR": "International Conference on Learning Representations",
            "ICML": "International Conference on Machine Learning",
            "AAAI": "Proceedings of the AAAI Conference on Artificial Intelligence",
            "IJCAI": "International Joint Conference on Artificial Intelligence",
            "CVPR": "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
            "ICCV": "Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)",
            "MM": "Proceedings of the ACM International Conference on Multimedia",
            "KDD": "Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
            "CIKM": "Proceedings of the ACM International Conference on Information and Knowledge Management",
            "SIGIR": "Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval",
            "WSDM": "Proceedings of the ACM International Conference on Web Search and Data Mining",
            "ECIR": "European Conference on Information Retrieval",
            "ECCV": "European Conference on Computer Vision",
            "COLT": "Conference on Learning Theory",
            "AISTATS": "International Conference on Artificial Intelligence and Statistics",
            "INTERSPEECH": "Conference of the International Speech Communication Association",
            "ISWC": "Proceedings of the ACM International Symposium on Wearable Computers",
            "JMLR": "Journal of Machine Learning Research",
            "VLDB": "International Conference on Very Large Data Bases",
            "ICME": "IEEE International Conference on Multimedia and Expo (ICME)",
            "TIP": "IEEE Transactions on Image Processing",
            "TPAMI": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
            "RECSYS": "Proceedings of the ACM Conference on Recommender Systems",
            "TKDE": "IEEE Transactions on Knowledge and Data Engineering",
            "TOIS": "ACM Transactions on Information Systems",
            "ICDM": "IEEE International Conference on Data Mining (ICDM)",
            "TASLP": "IEEE/ACM Transactions on Audio, Speech, and Language Processing",
            "BMVC": "British Machine Vision Conference Proceedings",
            "MICCAI": "Medical Image Computing and Computer Assisted Intervention",
            "IJCV": "International Journal of Computer Vision",
            "TNNLS": "IEEE Transactions on Neural Networks and Learning Systems",
            "FAST": "Proceedings of the USENIX Conference on File and Storage Technologies",
            "SIGMOD": "Proceedings of the ACM on Management of Data",
            "NIPS": "Advances in Neural Information Processing Systems",
            "MLSYS": "Proceedings of Machine Learning and Systems",
            "WACV": "Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)",
        }
        # try:
        #     paper_name = self.get_info_by_id(ID)['title']
        #     search_query = scholarly.search_pubs(paper_name)
        #     paper = next(search_query)
        #     bibtex = scholarly.bibtex(paper)
        # except:
        
        # otherwise, build the bibtex manually
        paper_info = self.get_info_by_id(ID)
        bib_id = "@inproceedings{"+paper_info["author"][0].split()[0].lower()+paper_info['year']+paper_info["title"].split()[0].lower()
        conf_name = paper_info["conference"]
        if conf_name in CONF_FULL_NAME:
            conf_name = CONF_FULL_NAME[conf_name]
        bibtex = f"{bib_id},\n\
        title={{{paper_info['title']}}},\n\
        author={{{' and '.join(paper_info['author'])}}},\n\
        year={{{paper_info['year']}}},\n\
        booktitle={{{conf_name}}}\n\
        "
        bibtex = bibtex+"}"
        return bibtex
    
    def get_content_by_id(self,ID):
        """
        get the content of a paper (text extracted from the pdf) by a given ID

        return a list of strings, each string is a page of the paper
        """
        found_paper_content = []
        if ID not in self.lstIDs:
            return found_paper_content
        
        for paper in self.all_papers_data:
            if paper["paper_id"] == ID:
                found_paper_content = paper["text_list"]
                break
        
        return found_paper_content

    def group_papers_by_LSA(self, num_groups=10, ngram_range=(2, 4), n_keywords=5, n_features=10000):
        """
        group papers into num_groups by XXX based on title+abstract (e.g., LSA, 
        pls replace XXX with specific method, you can implement multiple functions of this)
        this function returns a list of groups, each with a group name 
        and a brief discription (e.g., group name: xxx based methods, description: this group of methods uses ...)
        """
        
        documents = [paper['paper_name'] + " " + paper['paper_abstract'] for paper in self.all_papers_data]
    
        tfidf_vectorizer = TfidfVectorizer(max_df=0.65, max_features=n_features,
                                        min_df=2, stop_words='english',
                                        use_idf=True, ngram_range=ngram_range)
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

        svd_model = TruncatedSVD(n_components=num_groups, algorithm='randomized', n_iter=100, random_state=42)
        lsa = make_pipeline(svd_model, Normalizer(copy=False))
        
        dtm_lsa = lsa.fit_transform(tfidf_matrix)
        
        topics = []
        terms = tfidf_vectorizer.get_feature_names_out()
        for i, comp in enumerate(svd_model.components_):
            terms_comp = zip(terms, comp)
            sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:n_keywords]
            topic_keywords = [t[0] for t in sorted_terms] 
            topics.append(topic_keywords)
        
        return topics

class AIPaperRetriver:
    """
    pass in the target AIPaperCopus
    """
    def __init__(self, pAIPaperCopus, llm_model, llm_base, llm_key="EMPTY"):
        self.pAIPaperCopus = pAIPaperCopus
        self.device = torch.device("cuda")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.paper_embeds = []
        self.client = OpenAI(
            api_key = llm_key,
            base_url = llm_base
        )
        self.llm_model = llm_model

    def get_text_CLIP_embedding(self, text):
        inputs = self.processor(text=[text], padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        outputs = self.clip_model.get_text_features(**inputs)
        text_embeds = outputs / outputs.norm(dim=-1, keepdim=True)
        return text_embeds
    
    def search_by_CLIP_similarity(self, query, top_n=10):
        """
        search papers by CLIP similarity,
        return a ranked list of paper IDs based on the similarity of the query (a string) and the paper title+abstract
        the similariites should be returned as well
        """
        query_embeds = self.get_text_CLIP_embedding(query)
        
        top_papers_heap = []
        min_heap_size = 0

        for paper in tqdm(self.pAIPaperCopus.all_papers_data, desc="Get papers' embeddings"):
            paper_id = paper["paper_id"]
            paper_info = self.pAIPaperCopus.get_info_by_id(paper_id)
            paper_abs = paper_info["abstract"]
            paper_text = paper["paper_name"] + " " + paper_abs
            paper_embed = self.get_text_CLIP_embedding(paper_text)
            
            similarity = (query_embeds @ paper_embed.T).item()
            
            if min_heap_size < top_n:
                heapq.heappush(top_papers_heap, (similarity, paper_id))
                min_heap_size += 1
            else:
                heapq.heappushpop(top_papers_heap, (similarity, paper_id))
        
        top_papers_heap.sort(reverse=True, key=lambda x: x[0])
        ranked_paper_ids = [paper_id for _, paper_id in top_papers_heap]
        similarities = [sim for sim, _ in top_papers_heap]

        return ranked_paper_ids, similarities

    def getAnswerFromQuestion(self, question):
        """
        Return the answer from the question, not context-aware
        """
        if (len(question) == 0):
            return "Empty question."
    
        message = [{"role": "user", "content": question}]
        
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=message)
        
        reply = response.choices[0].message.content
        
        return reply
    
    def evaluate_relevance_by_LLM(self, query, value):
        prompt = f"You are a helpful assistant. \
            Please evaluate the relevance of the given paper to the given query. \
            If the paper is relevant to the query, please answer '1'. Otherwise, please answer '0'.\
            You can only answer '1' or '0', you are not allowed to provide any other information or explanation.\
            The query is: {query}. The paper title is: {value['title']}. The paper abstract is: {value['abstract']}."
        
        response = self.getAnswerFromQuestion(prompt)
        response = response.strip().replace("\n", "").replace(" ", "")
        if "1" in response:
            return True
        else:
            return False
    
    def search_by_LLM(self, query, limit=10):
        """
        search using LLM to evaluate the relevance of a paper to a given query and the paper title+abstract
        """
        found_paper_id = []
        for paper in tqdm(self.pAIPaperCopus.all_papers_data):
            paper_id = paper["paper_id"]
            paper_info = self.pAIPaperCopus.get_info_by_id(paper_id)
            if self.evaluate_relevance_by_LLM(query, {"title": paper_info["title"], "abstract": paper_info["abstract"]}):
                found_paper_id.append(paper_id)
                if len(found_paper_id) >= limit:
                    break
        return found_paper_id


if __name__ == "__main__":
    
    # 1. test on AIPaperCopus
    json_data_path = "/home/wengyu/work/AI-Paper-Collector/cache/sub_pdf_text_id"
    pAIPaperCopus = AIPaperCopus(json_data_path)
    ids = pAIPaperCopus.get_list_of_ids()
    all_paper = pAIPaperCopus.all_papers_data
    all_paper_by_conf = pAIPaperCopus.all_papers_data_by_conf
    print(len(ids))
    print(len(all_paper))
    print(all_paper_by_conf.keys())
    print(pAIPaperCopus.get_info_by_id(ids[10]))
    # print(pAIPaperCopus.get_content_by_id(ids[10]))
    print(pAIPaperCopus.get_bibtex_by_id(ids[10]))
    
    # 2. test on AIPaperRetriver
    llm_model = "llama3"
    llm_base = "http://158.132.255.128:11434/v1"   
    pAIPaperRetriver = AIPaperRetriver(pAIPaperCopus, llm_model=llm_model, llm_base=llm_base)
    
    query = "SIA-GCN"
    serach_by_clip_results = pAIPaperRetriver.search_by_CLIP_similarity(query, top_n=10)
    print(serach_by_clip_results[0], serach_by_clip_results[1])
    
    llm_serach_results = pAIPaperRetriver.search_by_LLM(query, limit=10)
    print(llm_serach_results)