class Tweet:
    """A class to hold tweet data"""
    def __init__(self, text, full_name, lat, lon, country=None, created_at=None):
        self.textAttr = text
        self.full_nameAttr = full_name
        self.latAttr = lat
        self.lonAttr = lon
        self.countryAttr = country
        self.created_atAttr = created_at

    def getText(self):
        return self.textAttr

    def getFullName(self):
        return self.full_nameAttr

    def getLat(self):
        return self.latAttr

    def getLon(self):
        return self.lonAttr
    
    def getCountry(self):
        return self.countryAttr

    def getCreatedAt(self):
        return self.created_atAttr
 
