import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)

matplotlib.rcParams['text.latex.preamble'] = [
#       r'\usepackage{helvet}',
       r'\usepackage{siunitx}',
       r'\sisetup{detect-all}',
       r'\usepackage{sansmath}',
       r'\sansmath'
]

sns.set_context('paper', font_scale=2.5)

