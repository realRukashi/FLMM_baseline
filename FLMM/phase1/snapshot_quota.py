from pathlib import Path
import json
from datetime import datetime,timezone
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate(); q=api.quota_view()
data={'observed_at_utc':datetime.now(timezone.utc).isoformat(),'refresh_at':str(q.quota_refresh_time)}
for name,item in [('gpu',q.gpu_quota),('tpu',q.tpu_quota)]:
    data[name]=None if item is None else {'used_seconds':item.time_used.total_seconds(),'total_seconds':item.total_time_allowed.total_seconds(),'remaining_seconds':max(0,(item.total_time_allowed-item.time_used).total_seconds())}
path=Path(__file__).resolve().parent/'results/quota_snapshot.json'; path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(data,indent=2),encoding='utf-8'); print(json.dumps(data,indent=2))
