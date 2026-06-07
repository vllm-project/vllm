from scheduler import Scheduler
from request import Request

def main():
    max_num_seqs = 5
    scheduler = Scheduler(max_num_seqs)

    # Create requests with different priorities
    request1 = Request(1)
    request2 = Request(2)
    request3 = Request(3)

    # Add requests to the scheduler
    scheduler.add_request(request1)
    scheduler.add_request(request2)
    scheduler.add_request(request3)

    # Schedule the requests
    scheduler.schedule()

    # Print the scheduled requests
    for request in scheduler.running_queue:
        print(request.priority)

if __name__ == "__main__":
    main()