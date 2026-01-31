mod generated;
mod openai;
mod grpc;

use grpc::VllmClient;

#[tokio::main]
async fn main() {
    println!("Modules compiled successfully");

    // Note: This would fail without a running server, just verifying types compile
    let _ = VllmClient::connect("localhost:50051");
}
