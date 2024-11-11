import re
import pandas as pd
import numpy as np
import spacy
from dframcy import DframCy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pprint
zh = spacy.load("zh_core_web_sm")


file_path = '/Users/r.a.n.d.y.w./Downloads/course-list.csv'
data = pd.read_csv(file_path,header=None)

course_list = data[0].apply(lambda x: re.sub(r'（[I|V|X|一二三四五六七八九十]+）', '', x)).tolist()
course_list = list(dict.fromkeys(course_list))


def vectorize(course_name):
    return zh(course_name).vector

def cosine_similarity(vec1, vec2):
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

course_vectors = list(zh.pipe(course_list))
vector_list = [doc.vector for doc in course_vectors]
similarity_matrix = np.zeros((len(course_list), len(course_list)))

for i, vec1 in enumerate(vector_list):
    for j in range(i, len(vector_list)):
        similarity = cosine_similarity(vec1, vector_list[j])
        similarity_matrix[i][j] = similarity
        similarity_matrix[j][i] = similarity

# Convert the matrix to a DataFrame for readability
similarity_df = pd.DataFrame(similarity_matrix, index=course_list, columns=course_list)
sorted_similarity_df = similarity_df.apply(lambda row: row.sort_values(ascending=False).index.tolist(), axis=1)



# Display the similarity matrix
pprint.pprint(sorted_similarity_df)

