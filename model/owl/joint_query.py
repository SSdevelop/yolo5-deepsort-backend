class Query:
    def __init__(self) -> None:
        self.json=''
        self.target=''
        self.query=''

class JointQuery:
    def __init__(self) -> None:
        self.query_list=[]
        self.video_results=[]
    
    def sort_top_k(self):
        pass

# select boxes with "white shirt" as white_shirt
# select boxes with "blue jeans" as blue_jeans
# select object
# select frames (blue_jeans, white shirt, man) where blue_jeans union white_shirt==man head 30