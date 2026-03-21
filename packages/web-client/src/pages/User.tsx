import { useParams } from "@solidjs/router";
import { createResource, Switch, Match } from "solid-js";
import { Layout } from "../layouts/Layout";
import axios, { AxiosError } from "axios";
import { UserInfo } from "../components/UserInfo";
import { useAuth } from "../auth";

export default function User() {
  const params = useParams();
  const { status: mine } = useAuth();
  const userId = Number(params.id);
  const [userInfo] = createResource(() =>
    axios.get(`users/${userId}`).then((res) => res.data),
  );
  return (
    <Layout>
      <Switch>
        <Match when={userInfo.loading}>正在加载中...</Match>
        <Match when={userInfo.error}>
          加载失败：{" "}
          {userInfo.error instanceof AxiosError
            ? userInfo.error.response?.data.message
            : userInfo.error}
        </Match>
        <Match when={userInfo()}>
          <div class="w-full flex flex-row justify-center">
            <UserInfo
              {...userInfo()}
              editable={userInfo()?.id === mine()?.id}
            />
          </div>
        </Match>
      </Switch>
    </Layout>
  );
}
