
import {
  Show,
  createResource,
  Switch,
  Match,
  For,
  createSignal,
  onMount,
  createEffect,
} from "solid-js";
import { Layout } from "../layouts/Layout";
import { A, useNavigate, useSearchParams } from "@solidjs/router";
import axios, { AxiosError } from "axios";
import { DeckBriefInfo } from "../components/DeckBriefInfo";
import { RoomDialog } from "../components/RoomDialog";
import { roomCodeToId } from "../utils";
import { RoomInfo } from "../components/RoomInfo";
import { useDecks } from "./Decks";
import { Login } from "../components/Login";
import { useAuth } from "../auth";

export default function Home() {
  const {
    status,
    loading: userLoading,
    error: userError,
    logout,
  } = useAuth();
  const navigate = useNavigate();
  const { decks, loading: decksLoading, error: decksError } = useDecks();

  const [roomCodeValid, setRoomCodeValid] = createSignal(false);
  let createRoomDialogEl!: HTMLDialogElement;
  let joinRoomDialogEl!: HTMLDialogElement;
  const [joiningRoomInfo, setJoiningRoomInfo] = createSignal<any>();

  const [currentRoom] = createResource(() =>
    axios.get("rooms/current").then((r) => r.data),
  );
  const [allRooms, { refetch: refreshAllRooms }] = createResource(() =>
    axios
      .get("rooms")
      .then((e) => e.data.filter((r: any) => r.id !== currentRoom()?.id)),
  );

  const isLogin = () => {
    const { type } = status();
    return type !== "notLogin";
  };

  const isOffline = () => {
    const { type } = status();
    return type === "offline";
  };

  const createRoom = () => {
    if (!decks().count) {
      alert("请先创建一组牌组");
      navigate("/decks/new");
      return;
    }
    createRoomDialogEl.showModal();
  };
  const joinRoomBySubmitCode = async (e: SubmitEvent) => {
    e.preventDefault();
    if (!decks().count) {
      alert("请先创建一组牌组");
      navigate("/decks/new");
      return;
    }
    const form = new FormData(e.target as HTMLFormElement);
    const roomCode = form.get("roomCode") as string;
    const roomId = roomCodeToId(roomCode);
    try {
      const { data } = await axios.get(`rooms/${roomId}`);
      setJoiningRoomInfo(data);
      joinRoomDialogEl.showModal();
    } catch (e) {
      if (e instanceof AxiosError) {
        alert(e.response?.data.message);
      }
      console.error(e);
      setJoiningRoomInfo();
    }
  };
  const joinRoomByInfo = (roomInfo: any) => {
    if (!decks().count) {
      alert("请先创建一组牌组");
      navigate("/decks/new");
      return;
    }
    setJoiningRoomInfo(roomInfo);
    joinRoomDialogEl.showModal();
  };

  return (
    <Layout>
      <div class="container mx-auto h-full">
        <Switch>
          <Match when={userLoading()}>
            <div class="text-gray-500">Loading now, please wait...</div>
          </Match>
          <Match when={userError()}>
            <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
              <div class="flex items-start">
                <div class="flex-shrink-0">
                  <span class="text-2xl">🚫</span>
                </div>
                <div class="ml-3 flex-1">
                  <h3 class="text-lg font-medium text-red-800 dark:text-red-200">
                    网络连接错误
                  </h3>
                  <div class="mt-2 text-red-700 dark:text-red-300">
                    <p class="text-sm">
                      无法连接到后端服务器。错误详情：
                    </p>
                    <p class="text-sm font-mono mt-1 p-2 bg-red-100 dark:bg-red-800 rounded">
                      {userError()?.message ?? String(userError())}
                    </p>
                  </div>
                  <div class="mt-4">
                    <div class="flex space-x-3">
                      <button
                        onClick={async () => {
                          try {
                            await logout();
                          } catch (e) {
                            // 登出可能也会失败，忽略错误
                            console.log("Forced logout completed");
                            // 强制刷新页面
                            window.location.reload();
                          }
                        }}
                        class="px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded-lg transition-colors"
                      >
                        🚪 退出登录并重试
                      </button>
                      <button
                        onClick={() => window.location.reload()}
                        class="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white text-sm font-medium rounded-lg transition-colors"
                      >
                        🔄 刷新页面
                      </button>
                    </div>
                    <div class="mt-3 text-sm text-red-600 dark:text-red-400">
                      <p>• 检查后端服务器是否正在运行</p>
                      <p>• 检查网络连接是否正常</p>
                      <p>• 可以尝试使用离线模式的AI对战功能</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Match>
          <Match when={isOffline()}>
            <div class="flex flex-col h-full min-h-0">
              <div class="flex-shrink-0 mb-8">
                <h2 class="text-3xl font-light">
                  🚫 离线模式 - 后端服务不可用
                </h2>
                <p class="text-gray-600 dark:text-gray-400 mt-2">
                  检测到网络连接问题，已启用离线模式。您可以：
                </p>
                <ul class="list-disc list-inside text-gray-600 dark:text-gray-400 mt-2 space-y-1">
                  <li>尝试刷新页面重新连接</li>
                  <li>检查后端服务是否正在运行</li>
                  <li>使用本地AI对战功能（无需后端）</li>
                </ul>
              </div>
              <div class="flex flex-grow flex-col-reverse md:flex-row gap-8 min-h-0">
                <div class="h-full w-full md:w-60 flex flex-col items-start md:bottom-opacity-gradient">
                  <A
                    href="/ai-battle"
                    class="text-xl font-bold text-blue-500 hover:underline mb-4"
                  >
                    🤖 AI对战 (离线可用)
                  </A>
                  <div class="text-sm text-gray-500 dark:text-gray-400">
                    <p>• 多种AI难度级别</p>
                    <p>• 实时对战体验</p>
                    <p>• 无需网络连接</p>
                  </div>
                </div>
                <div class="b-r-gray-200 b-1 hidden md:block" />
                <div class="flex-grow flex flex-col">
                  <h4 class="text-xl font-bold mb-5">离线功能</h4>
                  <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                      <h5 class="font-bold text-lg mb-2">🎮 AI对战</h5>
                      <p class="text-gray-600 dark:text-gray-400 text-sm mb-3">
                        与智能AI进行实时对战，支持多种难度级别和策略
                      </p>
                      <A
                        href="/ai-battle"
                        class="inline-block px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                      >
                        开始对战
                      </A>
                    </div>
                    <div class="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                      <h5 class="font-bold text-lg mb-2">🔧 故障排除</h5>
                      <p class="text-gray-600 dark:text-gray-400 text-sm mb-3">
                        尝试解决网络连接问题
                      </p>
                      <div class="space-y-2">
                        <button
                          onClick={() => window.location.reload()}
                          class="block w-full px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
                        >
                          🔄 刷新页面
                        </button>
                        <button
                          onClick={async () => {
                            try {
                              await logout();
                            } catch (e) {
                              console.log("Logout completed");
                            }
                          }}
                          class="block w-full px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                        >
                          🚪 退出登录
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Match>
          <Match when={isLogin()}>
            <div class="flex flex-col h-full min-h-0">
              <div class="flex-shrink-0 mb-8">
                <h2 class="text-3xl font-light">
                  {status().type === "guest" ? "游客 " : ""}
                  {status().name}，欢迎你！
                </h2>
              </div>
              <div class="flex flex-grow flex-col-reverse md:flex-row gap-8 min-h-0">
                <div class="h-full w-full md:w-60 flex flex-col items-start md:bottom-opacity-gradient">
                  <A
                    href="/decks"
                    class="text-xl font-bold text-blue-500 hover:underline mb-4"
                  >
                    我的牌组
                  </A>
                  <Switch>
                    <Match when={decksLoading()}>
                      <div class="text-gray-500">牌组信息加载中…</div>
                    </Match>
                    <Match when={decksError()}>
                      <div class="text-gray-500">
                        牌组信息加载失败：
                        {decksError()?.message ?? String(decksError())}
                      </div>
                    </Match>
                    <Match when={true}>
                      <div class="flex flex-row flex-wrap md:flex-col gap-2">
                        <For
                          each={decks().data}
                          fallback={
                            <div class="text-gray-500">
                              暂无牌组，
                              <A href="/decks/new" class="text-blue-500">
                                前往添加
                              </A>
                            </div>
                          }
                        >
                          {(deckData) => <DeckBriefInfo {...deckData} />}
                        </For>
                      </div>
                    </Match>
                  </Switch>
                </div>
                <div class="b-r-gray-200 b-1 hidden md:block" />
                <div class="flex-grow flex flex-col">
                  <h4 class="text-xl font-bold mb-5">开始游戏</h4>
                  <Show
                    when={!currentRoom()}
                    fallback={
                      <div class="mb-8">
                        <RoomInfo {...currentRoom()} />
                      </div>
                    }
                  >
                    <div class="flex flex-col md:flex-row gap-2 md:gap-5 items-center mb-8">
                      <button
                        class="flex-shrink-0 w-full md:w-35 btn btn-solid-green h-2.3rem"
                        onClick={createRoom}
                      >
                        创建房间…
                      </button>
                      <span class="flex-shrink-0">或者</span>
                      <A
                        href="/ai-battle"
                        class="flex-shrink-0 w-full md:w-35 btn btn-solid-purple h-2.3rem text-center"
                      >
                        🤖 AI对战
                      </A>
                      <span class="flex-shrink-0">或者</span>
                      <form
                        class="flex-grow flex flex-row w-full md:w-unset"
                        onSubmit={joinRoomBySubmitCode}
                      >
                        <input
                          type="text"
                          class="input input-solid rounded-r-0 b-r-0 flex-grow md:flex-grow-0 text-1rem line-height-none h-2.3rem"
                          name="roomCode"
                          placeholder="输入房间号"
                          inputmode="numeric"
                          pattern="\d{4}"
                          onInput={(e) =>
                            setRoomCodeValid(e.target.checkValidity())
                          }
                          autofocus
                          required
                        />
                        <button
                          type="submit"
                          class="flex-shrink-0 w-20 sm:w-35 btn btn-solid rounded-l-0 h-2.3rem"
                          disabled={!roomCodeValid()}
                        >
                          加入房间…
                        </button>
                      </form>
                    </div>
                  </Show>
                  <h4 class="text-xl font-bold mb-5 flex flex-row items-center gap-2">
                    当前对局
                    <button
                      class="btn btn-ghost-primary p-1"
                      onClick={refreshAllRooms}
                    >
                      <i class="i-mdi-refresh" />
                    </button>
                  </h4>
                  <ul class="flex gap-2 flex-row flex-wrap">
                    <Switch>
                      <Match when={allRooms.loading}>
                        <div class="text-gray-500">对局信息加载中…</div>
                      </Match>
                      <Match when={allRooms.error}>
                        <div class="text-red-500">
                          对局信息加载失败：
                          {allRooms.error instanceof AxiosError
                            ? allRooms.error.response?.data.message
                            : allRooms.error}
                        </div>
                      </Match>
                      <Match when={true}>
                        <For
                          each={allRooms()}
                          fallback={<div class="text-gray-500">暂无对局</div>}
                        >
                          {(roomInfo) => (
                            <li>
                              <RoomInfo {...roomInfo} onJoin={joinRoomByInfo} />
                            </li>
                          )}
                        </For>
                      </Match>
                    </Switch>
                  </ul>
                </div>
              </div>
              <RoomDialog ref={createRoomDialogEl!} />
              <RoomDialog
                ref={joinRoomDialogEl!}
                joiningRoomInfo={joiningRoomInfo()}
              />
            </div>
          </Match>
          <Match when={true}>
            <div class="w-full flex justify-center">
              <Login />
            </div>
          </Match>
        </Switch>
      </div>
    </Layout>
  );
}
