#!/usr/bin/env python3
from paramiko import SSHClient, WarningPolicy, SSHConfig, ProxyCommand
from scp import SCPClient    # from https://github.com/jbardin/scp.py
import sys
import os

from multiprocessing import Process, Queue
import threading


# Define default simple progress callback that prints the current percentage completed for the file
def simple_callback(filename, size, sent):
    sys.stdout.write("%s\'s progress: %.2f%%   \r" % (filename, float(sent)/float(size)*100) )


def scp_thread(scp, src_blob, dst_blob):  # putting and closing together in one
    scp.put(src_blob, dst_blob)
    scp.close()


def scp_upload(src_blob='Preproc.tar.gz', dst_blob="~", options={'hostname': 'lecun', 'username': 'shawley'}, progress=simple_callback):
    # from https://gist.github.com/acdha/6064215

    #--- Make the Paramiko SSH thing use my .ssh/config file (b/c I like ProxyCommand!)
    client = SSHClient()
    client.load_system_host_keys()
    client._policy = WarningPolicy()
    client.set_missing_host_key_policy(WarningPolicy())  # hmm. WarningPolicy? Most people use AutoAddPolicy. 

    ssh_config = SSHConfig()
    user_config_file = os.path.expanduser("~/.ssh/config")
    if os.path.exists(user_config_file):
        with open(user_config_file) as f:
            ssh_config.parse(f)

    cfg = {'hostname': options['hostname'], 'username': options["username"]}

    user_config = ssh_config.lookup(cfg['hostname'])
    for k in ('hostname', 'username', 'port'):
        if k in user_config:
            cfg[k] = user_config[k]

    if 'proxycommand' in user_config:
        cfg['sock'] = ProxyCommand(user_config['proxycommand'])

    client.connect(**cfg)

    socket_timeout = None # number of seconds for timeout. None = never timeout. TODO: None means program may hang. But timeouts are annoying!

    # SCPCLient takes a paramiko transport and progress callback as its arguments.
    scp = SCPClient(client.get_transport(), progress=progress, socket_timeout=socket_timeout)

    # NOW we can finally upload! (in a separate process)
    #scp.put(src_blob, dst_blob)   # now in scp_thread

    # we want this to be non-blocking so we stick it in a thread
    thread = threading.Thread(target=scp_thread, args=(scp,src_blob, dst_blob) )
    thread.start()

    #scp.close()  # now in scp_thread


if __name__ == '__main__':
    scp_upload()
