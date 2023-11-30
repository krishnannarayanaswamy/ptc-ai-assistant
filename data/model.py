class Product:
    def __init__(self, **kwargs):
        self.item_code = kwargs['item_code']
        self.item_name = kwargs['item_name']
        self.description = kwargs['description']
        self.price = kwargs['price']
        self.availability = kwargs['availability']
    