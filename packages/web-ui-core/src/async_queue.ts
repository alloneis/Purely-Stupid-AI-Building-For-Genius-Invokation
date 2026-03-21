type Task<T = unknown> = () => Promise<T>;

export class AsyncQueue {
  private queue: Task[] = [];
  private isProcessing: boolean = false;

  push<T>(task: Task<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const taskWithPromise = async () => {
        try {
          const ret = await task();
          resolve(ret);
        } catch (error) {
          reject(error);
        }
      };

      this.queue.push(taskWithPromise);
      if (!this.isProcessing) {
        this.processQueue();
      }
    });
  }

  private async processQueue(): Promise<void> {
    this.isProcessing = true;
    while (this.queue.length > 0) {
      const task = this.queue.shift()!;
      await task();
    }
    this.isProcessing = false;
  }
}
