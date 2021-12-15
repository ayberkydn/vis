import wandb

api = wandb.Api()

sweep = api.sweep("ayberkydn/vis/ffey02t9")

# for run in sweep.runs:
#     for file in run.files():
#         try:
#             file.download()
#         except:
#             pass

run = sweep.runs[2]
for file in run.files():
    file.download("asd")
