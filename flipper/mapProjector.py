class mapProjector( object ):

    def projectDataToMap( self, data, m ):
        m.data[:] += data.data[:]
        m.weight[:] += 1

    def projectMapToData( self, m, data):
        data.data[:] = m.data[:]
