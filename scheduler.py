import heapq

class Scheduler:
    def __init__(self, max_num_seqs):
        self.max_num_seqs = max_num_seqs
        self.running_queue = []
        self.waiting_queue = []

    def add_request(self, request):
        # Add request to waiting queue
        heapq.heappush(self.waiting_queue, (request.priority, request))

    def schedule(self):
        # Check if there are requests in the waiting queue that can be scheduled
        while self.waiting_queue and len(self.running_queue) < self.max_num_seqs:
            # Get the highest priority request from the waiting queue
            _, request = heapq.heappop(self.waiting_queue)

            # Check if the request can be scheduled
            if self.can_schedule(request):
                # Add the request to the running queue
                self.running_queue.append(request)
            else:
                # If the request cannot be scheduled, add it back to the waiting queue
                heapq.heappush(self.waiting_queue, (request.priority, request))

        # Check if there are requests in the running queue that can be preempted
        for i, running_request in enumerate(self.running_queue):
            # Check if there are requests in the waiting queue that have higher priority
            for waiting_request in self.waiting_queue:
                # Check if the waiting request has higher priority than the running request
                if waiting_request[0] > running_request.priority:
                    # Preempt the running request and add the waiting request to the running queue
                    self.running_queue[i] = waiting_request[1]
                    self.waiting_queue.remove(waiting_request)
                    break

    def can_schedule(self, request):
        # Check if the request can be scheduled based on the available resources
        # This method should be implemented based on the specific resource constraints
        pass