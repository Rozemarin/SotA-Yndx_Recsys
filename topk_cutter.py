import pandas as pd


class Cutter:
    def __init__(self, top_k=1000):
        self.top_k = top_k

    def _get_items_that_were_left(self, recommended_items):
        items = set()
        for lst in recommended_items:
            items.update(lst[:self.top_k])
        return items

    def cut_df(self, df, recommended_items):
        items = self._get_items_that_were_left(recommended_items)
        return df[df['item_id'].isin(items)]
    
    
if __name__ == '__main__':
    data = {
        'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        'item_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'rating': [5, 4, 3, 2, 5, 4, 3, 2, 5, 4],
        'timestamp': [1672531200, 1672617600, 1672704000, 1672790400, 1672876800, 
                  1672963200, 1673049600, 1673136000, 1673222400, 1673308800]
    }
    df = pd.DataFrame(data)
    
    # Предполагается, что каждый список соответсвует одному пользователю
    recommended_items = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [1, 3, 5, 7, 9],
        [2, 4, 6, 8, 10]
    ]
    cutter = Cutter(top_k=3)
    filtered_df = cutter.cut_df(df, recommended_items)