from room_business.room_instance import RoomInstance


class BusinessBase:
    def __init__(self):
        self.parent = None

    def init(self, parent: RoomInstance):
        self.parent = parent

    async def run(self):
        pass
    

    
    