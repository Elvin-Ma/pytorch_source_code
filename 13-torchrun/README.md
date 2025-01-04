# 0 torchrun
- /root/miniconda3/envs/pytorch2.5/bin/torchrun

```python
#!/root/miniconda3/envs/pytorch2.5/bin/python
# -*- coding: utf-8 -*-
import re
import sys
from torch.distributed.run import main
if __name__ == '__main__':
    # sys.argv = ['/root/miniconda3/envs/pytorch2.5/bin/torchrun', '--nproc_per_node=2', '--rdzv_backend', 'c10d', '--rdzv_endpoint=localhost:0', '--local-ranks-filter', '0', '--role', 'rank', '--tee', '3', 'train.py', '--job.config_file', './train_configs/llama2_7b.toml']
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
```

# 1 torch.distributed.run
- torchrun 多进程启动流程如下：<br>
![torchrun](images/torchrun.png)

