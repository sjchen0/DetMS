import numpy as np
import pandas as pd
test = np.array([[135,1,1400,14],[135,1,1400,16],[135,2,1400,34],[246,1,1400,15],[246,1,1400,23]])
df = pd.DataFrame(test, columns=['txID','addrID','btime','value'])
import ipdb; ipdb.set_trace()
