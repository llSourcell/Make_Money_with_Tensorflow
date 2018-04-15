import gevent
import numpy as np


def run_concurrent_requests(request_data, clients):
    """ Makes predictions from all clients concurrently.

        Arguments:
        request_data -- data that can be fed into all clients
        clients -- a list of PredictClient.predict functions

        Returns:
        A list of predictions from each client.
    """

    jobs = [gevent.spawn(c, request_data) for c in clients]
    gevent.joinall(jobs, timeout=10)

    return list(map(lambda x: np.array(x.value), jobs))
