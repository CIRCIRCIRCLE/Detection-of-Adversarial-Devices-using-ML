from easyfl.client import BaseClient
import copy
import logging

from easyfl.pb import server_service_pb2 as server_pb
from easyfl.pb import common_pb2 as common_pb
from easyfl.protocol import codec

logger = logging.getLogger(__name__)

class CustomizedClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(CustomizedClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        self.is_byz = False

    def set_byz(self, is_byz: bool = True):
        self.is_byz = is_byz

    def post_train(self):
        self.model.cpu()

    def construct_upload_request(self):
        """Construct client upload request for training updates and testing results.

        Returns:
            :obj:`UploadRequest`: The upload request defined in protobuf to unify local and remote operations.
        """
        logger.info(f"task_id: {self.conf.task_id}, round_id: {self.conf.round_id}, client_id: {self.cid}")
        data = codec.marshal(server_pb.Performance(accuracy=self.test_accuracy, loss=self.test_loss))
        typ = common_pb.DATA_TYPE_PERFORMANCE
        try:
            if self._is_train:
                data = codec.marshal(copy.deepcopy(self.compressed_model))
                typ = common_pb.DATA_TYPE_PARAMS
                data_size = self.train_data.size(self.cid)
            else:
                data_size = 1 if not self.test_data else self.test_data.size(self.cid)
        except KeyError:
            # When the datasize cannot be get from dataset, default to use equal aggregate
            data_size = 1

        m = self._tracker.get_client_metric().to_proto() if self._tracker else common_pb.ClientMetric()
        return server_pb.UploadRequest(
            task_id=str(self.conf.task_id),  # Ensure task_id is a string
            round_id=str(self.conf.round_id),  # Ensure round_id is a string
            client_id=str(self.cid),  # Ensure client_id is a string
            content=server_pb.UploadContent(
                data=data,
                type=typ,
                data_size=data_size,
                metric=m,
            ),
        )
