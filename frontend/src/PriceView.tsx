import { useState } from 'react'
import type { KirimanDeteksi } from './DetectionView'
import {
  HARGA_SNACK,
  daftarGramasi,
  formatRupiah,
  hargaVarian,
  kemasanVarian,
  labelSnack,
} from './hargaSnack'

// varian: gramasi (g) -> qty. Satu snack bisa punya beberapa gramasi sekaligus.
type Item = { id: number; nama: string; varian: Record<number, number> }

let idBerikut = 1

// Membuat item baru tanpa varian; gramasi dipilih user setelah snack dipilih.
function buatItem(nama: string): Item {
  return { id: idBerikut++, nama, varian: {} }
}

// Mengelompokkan hasil deteksi per nama snack (satu item per jenis snack).
// Dari halaman deteksi (pilihTerendah), snack berkalori terendah langsung dipilih
// dengan gramasi terkecil x1. Dari menu Estimasi Harga, tidak ada pilihan awal.
function itemDariDeteksi(kiriman: KirimanDeteksi): Item[] {
  const valid = kiriman.deteksi.filter(d => HARGA_SNACK[d.name])
  const terendah = valid.length ? Math.min(...valid.map(d => d.calories)) : 0
  const nama = [...new Set(valid.map(d => d.name))]
  return nama.map(n => {
    const item = buatItem(n)
    const dipilih = kiriman.pilihTerendah
      && valid.some(d => d.name === n && d.calories === terendah)
    if (dipilih) item.varian[daftarGramasi(n)[0]] = 1
    return item
  })
}
// Menghitung subtotal seluruh varian pada satu item.
function subtotalItem(item: Item): number {
  return Object.entries(item.varian).reduce(
    (jumlah, [gramasi, qty]) => jumlah + hargaVarian(item.nama, Number(gramasi)) * qty,
    0,
  )
}

type Props = { kiriman: KirimanDeteksi; onGantiFoto: () => void }

// Menampilkan estimasi harga belanja dari snack yang terdeteksi.
export default function PriceView({ kiriman, onGantiFoto }: Props) {
  const [items, setItems] = useState<Item[]>(() => itemDariDeteksi(kiriman))
  // Beberapa snack boleh terbuka sekaligus; awalnya hanya yang sudah punya pilihan.
  const [terbuka, setTerbuka] = useState<number[]>(() =>
    items.filter(i => Object.keys(i.varian).length > 0).map(i => i.id),
  )

  // Membuka/menutup pilihan gramasi sebuah snack.
  function toggleBuka(id: number) {
    setTerbuka(terbuka.includes(id) ? terbuka.filter(t => t !== id) : [...terbuka, id])
  }

  // Memilih atau membatalkan satu gramasi; qty awal 1.
  function toggleGramasi(id: number, gramasi: number) {
    setItems(items.map(item => {
      if (item.id !== id) return item
      const varian = { ...item.varian }
      if (varian[gramasi]) delete varian[gramasi]
      else varian[gramasi] = 1
      return { ...item, varian }
    }))
  }

  // Mengubah qty satu gramasi (minimal 1).
  function ubahQty(id: number, gramasi: number, qty: number) {
    if (qty < 1) return
    setItems(items.map(item =>
      item.id === id ? { ...item, varian: { ...item.varian, [gramasi]: qty } } : item,
    ))
  }

  // Menambahkan snack secara manual dari dropdown dan langsung membukanya.
  function tambah(nama: string) {
    if (!nama) return
    const baru = buatItem(nama)
    setItems([...items, baru])
    setTerbuka([...terbuka, baru.id])
  }

  // Menghapus satu snack dari daftar.
  function hapus(id: number) {
    setItems(items.filter(i => i.id !== id))
    setTerbuka(terbuka.filter(t => t !== id))
  }

  const namaTerbuka = new Set(items.filter(i => terbuka.includes(i.id)).map(i => i.nama))
  const total = items.reduce((jumlah, i) => jumlah + subtotalItem(i), 0)
  const jumlahItem = items.reduce(
    (n, i) => n + Object.values(i.varian).reduce((a, b) => a + b, 0),
    0,
  )

  return (
    <div className="price">
      <div className="price-left">
        <div className="box">
          <img src={kiriman.gambar} />
          {kiriman.deteksi.map((d, indeks) => (
            <div
              key={indeks}
              onClick={() => {
                const item = items.find(i => i.nama === d.name)
                if (item) toggleBuka(item.id)
              }}
              className={'bbox' + (namaTerbuka.has(d.name) ? ' sel' : '')}
              style={{
                top: `${d.box[0] * 100}%`,
                left: `${d.box[1] * 100}%`,
                height: `${(d.box[2] - d.box[0]) * 100}%`,
                width: `${(d.box[3] - d.box[1]) * 100}%`,
              }}
            >
              <span>{labelSnack(d.name)}</span>
            </div>
          ))}
        </div>
        <div className="row">
          <button onClick={onGantiFoto}>GANTI FOTO</button>
        </div>
      </div>

      <div className="price-right">
        <p className="hint">
          👉 Klik snack di daftar (atau di foto), lalu pilih satu atau beberapa gramasi.
        </p>
        {items.length === 0 && <p>Belum ada snack. Tambahkan lewat dropdown di bawah.</p>}
        <div className="price-list">
          {items.map(item => {
            const aktif = terbuka.includes(item.id)
            return (
              <div
                className={'price-item' + (aktif ? ' sel' : '')}
                key={item.id}
                onClick={() => toggleBuka(item.id)}
              >
                <div className="price-head">
                  <b>{labelSnack(item.nama)}</b>
                  <b
                    className="del"
                    title="Hapus"
                    onClick={e => {
                      e.stopPropagation()
                      hapus(item.id)
                    }}
                  >
                    🗑️
                  </b>
                </div>
                {aktif && (
                  <div className="chips" onClick={e => e.stopPropagation()}>
                    {daftarGramasi(item.nama).map(g => (
                      <button
                        key={g}
                        className={'gram' + (item.varian[g] ? ' active' : '')}
                        onClick={() => toggleGramasi(item.id, g)}
                      >
                        {g} g {kemasanVarian(item.nama, g)} · {formatRupiah(hargaVarian(item.nama, g))}
                      </button>
                    ))}
                  </div>
                )}
                {Object.keys(item.varian).length === 0 && (
                  <p className="empty-var">Belum pilih gramasi</p>
                )}
                {Object.entries(item.varian)
                  .sort(([a], [b]) => Number(a) - Number(b))
                  .map(([gramasi, qty]) => {
                    const g = Number(gramasi)
                    return (
                      <div className="price-foot" key={g} onClick={e => e.stopPropagation()}>
                        <div className="qty">
                          <button disabled={qty <= 1} onClick={() => ubahQty(item.id, g, qty - 1)}>
                            −
                          </button>
                          <span>{qty}</span>
                          <button onClick={() => ubahQty(item.id, g, qty + 1)}>+</button>
                          <span>× {g} g</span>
                        </div>
                        <b>{formatRupiah(hargaVarian(item.nama, g) * qty)}</b>
                      </div>
                    )
                  })}
              </div>
            )
          })}
        </div>

        <div className="add-snack">
          <select value="" onChange={e => tambah(e.target.value)}>
            <option value="">+ Tambah snack lain...</option>
            {Object.keys(HARGA_SNACK).map(nama => (
              <option key={nama} value={nama}>{labelSnack(nama)}</option>
            ))}
          </select>
        </div>

        <div className="card summary">
          <p>Total estimasi ({jumlahItem} item)</p>
          <h3>{formatRupiah(total)}</h3>
        </div>
        <div className="note">
          <p>ℹ️ <b>Info:</b> Harga bersifat estimasi dan bisa berbeda di tiap toko.</p>
        </div>
      </div>
    </div>
  )
}
