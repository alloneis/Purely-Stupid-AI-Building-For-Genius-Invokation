import { Layout } from "../layouts/Layout";

export default function NotFound() {
  return (
    <Layout>
      <div class="flex flex-col items-center justify-center h-full">
        <h1 class="text-4xl font-bold">404 Not Found</h1>
        <p class="text-lg">页面未找到</p>
      </div>
    </Layout>
  );
}
