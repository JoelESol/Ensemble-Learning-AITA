import dataset.dataset as data
from networks.agent import run_all_agents
from networks.ensemble import VerdictEnsembleTransformer


results = run_all_agents(text)

output = data.get_dataset(1, 4)
print(len(output[0]))


