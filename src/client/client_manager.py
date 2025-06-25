import httpx
import asyncio

from pathlib import Path
from fake_useragent import FakeUserAgent
from tqdm.asyncio import tqdm_asyncio as tqdm

from src.config.config import Config
from src.utils.log_manager import log_manager


class ClientManager:
    """Менеджер для управления HTTP-клиентами."""

    def __init__(
        self,
        proxy_file: Path | str | None = None,
        timeout: httpx.Timeout = httpx.Timeout(
            connect=10.0, read=60.0, write=10.0, pool=10.0
        ),
        retries: int = 3,
        count_clients: int = 100,
    ):
        """
        Инициализация менеджера клиентов.

        Args:
            proxy_file (Path | str | None): Путь к файлу с прокси-серверами.
            timeout (httpx.Timeout): Таймаут для HTTP-запросов.
            retries (int): Количество попыток повторного запроса при ошибке.
            count_clients (int): Количество клиентов для создания если не указаны прокси.
        """

        self.proxy_file = Config.BASE_DIR / proxy_file if proxy_file else None
        self.timeout = timeout
        self.retries = retries
        self.count_clients = count_clients

        self.clients = []
        self.lock = asyncio.Lock()
        self.logger = log_manager.get_logger(__name__)

    async def setup(self) -> None:
        """Инициализация клиентов."""

        try:
            self.clients = await self._spawn_clients()
        except asyncio.CancelledError:
            self.logger.warning("Инициализация клиентов была отменена")

    async def pop_client(self) -> httpx.AsyncClient:
        """
        Извлечение клиента из пула клиентов.
        Если клиентов меньше 10, создаются новые клиенты.

        Returns:
            httpx.AsyncClient: HTTP-клиент для выполнения запросов.

        Raises:
            ValueError: Если нет доступных клиентов или не удалось создать новых.
        """

        async with self.lock:
            if not self.clients:
                raise ValueError("Нет доступных клиентов, проверьте настройки прокси")
            if len(self.clients) < 10:
                self.logger.debug(f"Осталось менее 10 клиентов, создаю новых")
                self.clients.extend(await self._spawn_clients(10))
            return self.clients.pop(0)

    async def _spawn_clients(self) -> list[httpx.AsyncClient]:
        """
        Создание клиентов с учетом прокси.

        Returns:
            list[httpx.AsyncClient]: Список созданных HTTP-клиентов.
        """

        fake_useragent = FakeUserAgent()
        http_transport = httpx.AsyncHTTPTransport(retries=self.retries)

        if not self.proxy_file or not self.proxy_file.exists():
            self.logger.warning("Прокси файл не указан или не существует")
            self.logger.warning(f"Создаю {self.count_clients} клиентов без прокси")
            return [
                self._generate_client(
                    fake_useragent,
                    http_transport,
                )
                for _ in range(self.count_clients)
            ]

        proxies = await self._read_http_proxies()

        if not proxies:
            self.logger.warning(f"В файле {self.proxy_file} не найдено рабочих прокси")
            self.logger.warning(f"Создаю {self.count_clients} клиентов без прокси")
            return [
                self._generate_client(
                    fake_useragent,
                    http_transport,
                )
                for _ in range(self.count_clients)
            ]

        self.logger.info(
            f"Доступно {len(proxies)} прокси. Создаю {len(proxies)} клиентов с прокси"
        )
        return [
            self._generate_client(
                fake_useragent,
                http_transport,
                self.retries,
                proxy,
            )
            for proxy in proxies
        ]

    def _generate_client(
        self,
        fake_useragent: FakeUserAgent,
        http_transport: httpx.AsyncHTTPTransport,
        retries: int | None = None,
        proxy: httpx.Proxy | None = None,
    ) -> httpx.AsyncClient:
        """
        Генерация HTTP-клиента с заданными параметрами.

        Args:
            fake_useragent (FakeUserAgent): Экземпляр для генерации User-Agent.
            http_transport (httpx.AsyncHTTPTransport): Транспорт для HTTP-клиента.
            retries (int | None): Количество попыток повторного запроса при ошибке.
            proxy (httpx.Proxy | None): Прокси-сервер для использования клиентом.

        Returns:
            httpx.AsyncClient: Сгенерированный HTTP-клиент.
        """

        if proxy:
            transport = httpx.AsyncHTTPTransport(retries=retries, proxy=proxy)
            self.logger.debug(f"Создан клиент с использованием прокси: {proxy}")
        else:
            transport = http_transport

        return httpx.AsyncClient(
            headers={"User-Agent": fake_useragent.random},
            transport=transport,
            timeout=self.timeout,
        )

    async def _read_http_proxies(self) -> list[httpx.Proxy]:
        """
        Чтение и проверка HTTP-прокси из файла.

        Returns:
            list[httpx.Proxy]: Список проверенных HTTP-прокси.
        """

        tasks = []
        proxies = []
        working_proxies = []

        with self.proxy_file.open("r", encoding="utf-8") as f:
            for line in f:
                host, port, user, password = line.strip().split(":")
                proxy = httpx.Proxy(url=f"http://{host}:{port}", auth=(user, password))
                proxies.append(proxy)
                tasks.append(self._check_http_proxy(proxy))

        self.logger.info(f"Проверка {len(proxies)} прокси серверов:")
        results = await tqdm.gather(*tasks, desc="Проверка прокси")

        working_count = 0
        for proxy, is_working in zip(proxies, results):
            if is_working:
                working_count += 1
                working_proxies.append(proxy)

        self.logger.info(f"Найдено рабочих прокси: {working_count} из {len(proxies)}")
        return working_proxies

    async def _check_http_proxy(
        self,
        proxy: httpx.Proxy,
        test_url: str = "https://httpbin.org/ip",
        retries: int = 1,
        timeout: float = 10.0,
    ) -> bool:
        """
        Проверка работоспособности HTTP-прокси.

        Args:
            proxy (httpx.Proxy): Прокси для проверки.
            test_url (str): URL для тестирования прокси.
            retries (int): Количество попыток повторного запроса.
            timeout (float): Таймаут для запроса.

        Returns:
            bool: True, если прокси работает, иначе False.
        """

        fake_useragent = FakeUserAgent()
        http_transport = httpx.AsyncHTTPTransport(retries=retries, proxy=proxy)
        http_timeout = httpx.Timeout(timeout)
        client = httpx.AsyncClient(
            headers={"User-Agent": fake_useragent.random},
            transport=http_transport,
            timeout=http_timeout,
        )

        async with client:
            try:
                response = await client.get(test_url)
                self.logger.debug(
                    f"Прокси {proxy.url} вернул статус {response.status_code}"
                )
                return response.status_code == 200

            except httpx.ProxyError:
                self.logger.warning(f"Прокси {proxy.url} не доступен")
                return False
            except httpx.TimeoutException:
                self.logger.warning(f"Прокси {proxy.url} истекло время ожидания")
                return False
            except httpx.RequestError as e:
                self.logger.warning(
                    f"Ошибка запроса через прокси {proxy.url}: {str(e)}"
                )
                return False
            except Exception as e:
                self.logger.debug(f"Ошибка при проверке прокси {proxy.url}: {str(e)}")
                return False
