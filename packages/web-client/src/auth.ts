
import { Accessor, createResource } from "solid-js";
import { GuestInfo, useGuestInfo } from "./guest";
import axios from "axios";

export interface UserInfo {
  type: "user";
  id: number;
  login: string;
  name?: string;
  chessboardColor: string | null;
}

export interface OfflineInfo {
  type: "offline";
  name: string;
  id: null;
  chessboardColor: null;
}

const NOT_LOGIN = {
  type: "notLogin",
  name: "",
  id: null,
  chessboardColor: null,
} as const;

type NotLogin = typeof NOT_LOGIN;

type AuthStatus = UserInfo | GuestInfo | OfflineInfo | NotLogin;

export interface UpdateInfoPatch {
  chessboardColor?: string | null;
}

export interface Auth {
  readonly status: Accessor<AuthStatus>;
  readonly loading: Accessor<boolean>;
  readonly error: Accessor<any>;
  readonly refresh: () => Promise<void>;
  readonly loginGuest: (name: string) => void;
  readonly setGuestId: (id: string) => void;
  readonly updateInfo: (patch: UpdateInfoPatch) => Promise<void>;
  readonly logout: () => Promise<void>;
}

const [user, { refetch: refetchUser }] = createResource<UserInfo | NotLogin>(
  () =>
    axios.get<UserInfo>("users/me")
      .then(({ data }) =>
        data
          ? {
              ...data,
              type: "user" as const,
              name: data.name ?? data.login,
            }
          : NOT_LOGIN
      )
      .catch((error) => {
        // 如果是网络错误或401错误，认为是未登录状态
        if (error.code === 'NETWORK_ERROR' ||
            error.response?.status === 401 ||
            error.message?.includes('Network Error')) {
          return NOT_LOGIN;
        }
        // 其他错误继续抛出
        throw error;
      })
);

const updateUserInfo = async (newInfo: Partial<UserInfo>) => {
  await axios.patch("users/me", newInfo);
};

export const useAuth = (): Auth => {
  const [guestInfo, setGuestInfo] = useGuestInfo();
  return {
    status: () => {
      const guest = guestInfo();
      if (guest) {
        return guest;
      }

      // 如果用户信息加载出错，提供离线模式
      if (user.state === "errored") {
        return {
          type: "offline",
          name: "离线模式",
          id: null,
          chessboardColor: null,
        } as const;
      }

      return (
        (user.state === "ready" || user.state === "refreshing")
          ? user()
          : NOT_LOGIN
      );
    },
    loading: () => guestInfo() === null && user.loading,
    error: () => (guestInfo() === null ? user.error : void 0),
    refresh: async () => {
      try {
        await refetchUser();
      } catch (error) {
        // 如果网络错误，静默处理，不抛出错误
        console.warn("Failed to refresh user info:", error);
      }
    },
    loginGuest: async (name: string) => {
      setGuestInfo({
        type: "guest",
        name,
        id: null,
        chessboardColor: null,
      });
    },
    setGuestId: (id: string) => {
      setGuestInfo(
        (oldInfo) =>
          oldInfo && {
            ...oldInfo,
            id,
          }
      );
    },
    updateInfo: async (patch) => {
      const guest = guestInfo();
      if (guest) {
        setGuestInfo({ ...guest, ...patch });
      } else {
        await updateUserInfo(patch);
        await refetchUser();
      }
    },
    logout: async () => {
      // 清除认证信息
      localStorage.removeItem("accessToken");
      setGuestInfo(null);

      // 强制刷新用户信息（这会触发重新获取，但由于token已清除，会返回NOT_LOGIN）
      try {
        await refetchUser();
      } catch (error) {
        // 如果网络错误，忽略，继续登出流程
        console.log("Logout completed (network error ignored)");
      }
    },
  };
};
