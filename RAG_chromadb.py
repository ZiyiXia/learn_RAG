import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import ollama


HF_token = ""

corpus = [
    "Cheli: A downtown Chinese restaurant presents a distinctive dining experience with authentic and sophisticated flavors of Shanghai cuisine. Price: $40-50",
    "Masa: Midtown Japanese restaurant with exquisite sushi and omakase experiences crafted by renowned chef Masayoshi Takayama. The restaurant offers a luxurious dining atmosphere with a focus on the freshest ingredients and exceptional culinary artistry. Avg cost: $500-600",
    "Per Se: A midtown restaurant features daily nine-course tasting menu and a nine-course vegetable tasting menu using classic French technique and the finest quality ingredients available. Avg cost: $300-400",
    "Ortomare: A casual, earthy Italian restaurant locates uptown, offering wood-fired pizza, delicious pasta, wine & spirits & outdoor seating. Avg cost: $30-50",
    "Banh: Relaxed, narrow restaurant in uptown, offering Vietnamese cuisine & sandwiches, famous for its pho and Vietnam sandwich. Avg cost: $20-30",
    "Living Thai: An uptown typical Thai cuisine with different kinds of curry, Tom Yum, fried rice, Thai ice tea, etc. Avg cost: $20-30",
    "Chick-fil-A: A Fast food restaurant with great chicken sandwich, fried chicken, fries, and salad, which can be found everywhere in New York. Avg cost: 10-20",
    "Joe's Pizza: Most famous New York pizza locates midtown, serving different flavors including classic pepperoni, cheese, spinach, and also innovative pizza. Avg cost: $15-25",
    "Red Lobster: In midtown, Red Lobster is a lively chain restaurant serving American seafood standards amid New England-themed decor, with fair price lobsters, shrips and crabs. Avg cost: $30-50",
    "Bourbon Steak: It accomplishes all the traditions expected from a steakhouse, offering the finest cuts of premium beef and seafood complimented by wine and spirits program. Avg cost: $100-150",
    "Da Long Yi: Locates in downtown, Da Long Yi is a Chinese Szechuan spicy hotpot restaurant that serves good quality meats. Avg cost: $30-50",
    "Mitr Thai: An exquisite midtown Thai restaurant with traditional dishes as well as creative dishes, with a wonderful bar serving cocktails. Avg cost: $40-60",
    "Yichiran Ramen: Famous Japenese ramen restaurant in both midtown and downtown, serving ramen that can be designed by customers themselves. Avg cost: $20-40",
    "BCD Tofu House: Located in midtown, it's famous for its comforting and flavorful soondubu jjigae (soft tofu stew) and a variety of authentic Korean dishes. Avg cost: $30-50",
]

user_input = "I want some Chinese food"

client = chromadb.PersistentClient(path=".")

bge_embedding = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=HF_token,
    model_name="BAAI/bge-large-en-v1.5"
)

# collection = client.create_collection(
#     name="my_collection", 
#     embedding_function=bge_embedding
# )

# collection.add(
#     documents=corpus,
#     ids=['id'+str(i) for i in range(len(corpus))]
# )

collection = client.get_collection(
    name="my_collection", 
    embedding_function=bge_embedding
)

res = collection.query(
    query_texts=[user_input], 
    n_results=3,
    include=["documents"]
)

# response = ollama.chat(model='llama3', messages=[
#     {
#         'role': 'user',
#         'content': 'Why is the sky blue?'
#     }
# ])
# print(response['message']['content'], 'end\n')
 
prompt="""
You are a bot that makes recommendations for restaurants. 
Please be brief, consie, and complete, answer in short sentences without extra information.
These are the restaurants list:
{recommended_activities}
The user's preference is: {user_input}
Provide the user with 2 recommended restaurants based on the user's preference.
"""

text = ollama.generate(model='llama3', prompt=prompt.format(user_input=user_input, recommended_activities=res['documents']))
print(text['response'])